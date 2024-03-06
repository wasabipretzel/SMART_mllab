# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import sys
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from itertools import accumulate
from transformers import AutoTokenizer

import requests
from PIL import Image
from io import BytesIO
import re
import numpy as np

#main model class
from llava.model.sequence_mm import SequentialMM_Model
# from llava.model.multimodal_projector.builder import build_vision_projector

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    query_num: Optional[int] = field(default=16)
    mm_projector_model_path: Optional[str] = field(default='/SeqMMLearning/checkpoints/llava-v1.5-7b/mm_projector.bin')

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    train_txt_path: str = '/data/dataset/split/train.txt'
    val_txt_path: str = '/data/dataset/split/val.txt'
    feature_path: str = '/data/dataset/features'

    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    qformer_lr: Optional[float] = None

    group_by_modality_length: bool = field(default=False)
    freeze_pretrained: bool=field(default=False)




def maybe_zero_3(param, ignore_status=False, name=None):
    param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_assembly(
    sources: Sequence[str],
    has_image: bool = False
) -> Dict:
    """
        prompt + Question : question + Answer : answer 
        가 있을 때 text_input, text_output을 return 

        text_input = 'prompt + Question : question + Answer :'
        text_output = 'answer'
    """
    #prompt : <im_st> <im_end> sys ### question ### answer
    header = f"{conversation_lib.default_conversation.system}\n\n" # general prompt 
    # conversation = _add_speaker_and_signal_assembly(header, sources) # 1. Add signal
    conversation = header + sources
    # conversations = [conversation]
    question_sep = "Question"
    answer_sep = "Answer"

    #text_input
    text_input = conversation.split(answer_sep)[0]

    text_output = answer_sep + conversation.split(answer_sep)[-1]

    # if text_output[0] != ':':
    #     text_input + ' : '
    # else: # ':' 가 이미 붙어 있는 경우 input 마지막에 'Answer ' 가 되도록 하여 full text가  'Answer : step 9 ~' 이렇게 되도록 한다
    #     text_input + ' '
    
    result = {
        "text_input" : text_input, #str
        "text_output" : text_output
    }

    return result

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LazySupervisedDataset_Assembly(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str, 
                 txt_path: str,
                 data_args: DataArguments,
                 is_train: bool):
        super(LazySupervisedDataset_Assembly, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # txt_file = open(txt_path, "r")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.all_data = list_data_dict
        self.txt_path = txt_path
        self.data_args = data_args
        self.feature_path = self.data_args.feature_path
        self.is_train=is_train

        with open(self.txt_path, 'r') as f:
            candidates = f.readlines() #[11, 32, 3, 0, ...] #학습 또는 val 에 사용할 후보군들
        self.candidates = [idx.strip() for idx in candidates]

    def __len__(self):
        return len(self.candidates)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
            input_ids : List[] -> sequence length
            qformer_input_ids : List[] -> question length
            target 
            "im_st" (input_ids)
            "im_end" (input_ids)
            "feat" [num_img, 576]
            "num_img" : int
        """
        #load text feature
        search_key = self.candidates[idx]
        single_case = self.all_data[search_key] #{id : ~, content : ~}
        serial_number = single_case["id"]
        content = single_case["content"]
 
        # input_ids, qformer_input_ids, target, im_st, im_end
        data_dict = preprocess_assembly(
            content,    
            has_image=False
        ) #input_ids with chunks


        #load image feature
        single_feature_path = os.path.join(self.feature_path, serial_number+'.npy')
        feat = torch.tensor(np.load(single_feature_path)) #[num_img, 576]
        num_img = feat.shape[0]
        img_getitem = {
            "feat" : feat, #tensor [num_img, 576]
            "num_img" : num_img, #int
            
        }

        data_dict.update(img_getitem)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #text
        b_text_input = []
        b_text_output = []
        #image
        b_feat = []
        b_img_nums = []
        #get largest image num in batch
        #NOTE text input / output 에 대한 input_ids 및 att은 model forward에서 생성
        for each_batch in instances:
            b_img_nums.append(each_batch["num_img"])
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])
            b_feat.append(each_batch["feat"])
        b_max_img = max(b_img_nums)


        result = {
            "text_input" : b_text_input,
            "text_output" : b_text_output,

            "image_feat" : b_feat,
            "image_num_in_batch" : b_img_nums
        
        }

        return result


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset_Assembly(
                                data_path=data_args.data_path,
                                txt_path=data_args.train_txt_path,
                                data_args=data_args, is_train=True)
    val_dataset =  LazySupervisedDataset_Assembly(
                                data_path=data_args.data_path,
                                txt_path=data_args.val_txt_path,
                                data_args=data_args, is_train=False)

    data_collator = DataCollatorForSupervisedDataset()
    
    
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower == 'None':
        model_args.vision_tower=None
    # if training_args.report_to == 'None':
        # training_args.report_to=None
        # training_args.pop('report_to')
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    
    llm = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )

    print(f"llm device : {llm.device}")
    
    print("initializing")
    #main model initialize
    model = SequentialMM_Model(llm=llm, query_num=model_args.query_num, args=model_args, device=training_args.device).to(training_args.device)
    print("model finished")
    model.load_mm_projector_state_dict()
    training_args.gradient_checkpointing=False
    model.llm.gradient_checkpointing_enable()
    model.llm.enable_input_require_grads()
    llm.config.use_cache = False


    ## PEFT(Parameter-Efficient Fine Tuning)   
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model.llm),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.llm.to(torch.bfloat16)
            if training_args.fp16:
                model.llm.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model.llm = get_peft_model(model.llm, lora_config) # LlavaLlamaForCausalLM -> PeftModelForCausalLM 모델 변경

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=llm,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        '''
        Prompt 처리 #NOTE 여기 어떻게 들어옴?
        '''
        model.llm_tokenizer.pad_token = model.llm_tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    

    print(f"lora params : {model.llm.print_trainable_parameters()}")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params : {count_parameters(model)}")

    ## DataLoader
    data_module = make_supervised_data_module(data_args=data_args)

    trainer = LLaVATrainer(model=model,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("tldr-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.llm.config.save_pretrained(training_args.output_dir)
            model.llm.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


# if __name__ == "__main__":
#     train()