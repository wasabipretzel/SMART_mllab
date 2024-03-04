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
    use_qformer: bool = True
    query_num: Optional[int] = field(default=16)
    mm_projector_model_path: Optional[str] = field(default='/SeqMMLearning/checkpoints/llava-v1.5-7b/mm_projector.bin')
    use_pretrained_qformer: bool = field(default=False)
    pretrained_qformer_path: str = field(default="/data/pretrained_models/qformer_pretrained")
    pretrained_qformer_tokenizer_path: str = field(default="/data/pretrained_models/qformer_pretrained/qformer_tokenizer")
    pretrained_qformer_query_token_path: str = field(default="/data/pretrained_models/qformer_pretrained/query_tokens/query_tokens.pth")


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
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
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



def _add_speaker_and_signal_assembly(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    conversation = "<im_st> <im_end> " + conversation
    source = " ### " + source
    source = source.replace("Answer", " ### Answer")
    conversation = conversation + source 

    return conversation


def preprocess_assembly(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    qf_tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    #prompt : <im_st> <im_end> sys ### question ### answer
    header = f"{conversation_lib.default_conversation.system}\n\n" # general prompt 
    conversation = _add_speaker_and_signal_assembly(header, sources) # 1. Add signal
    conversations = [conversation]

    IMAGE_START = '<im_st>'
    IMAGE_END = '<im_end>'
    SEPERATOR = '###'
    
    #question을 먼저 얻어야함
    question = conversation.split(SEPERATOR)[1].strip()
    text_source = conversation.split(IMAGE_START + ' ' + IMAGE_END)[-1] # ' sys ### question ### answer'
    #before_answer을 구하기 위함
    sys = text_source.split(SEPERATOR)[0].split(IMAGE_END)[-1] # ' sys'
    ques = text_source.split(SEPERATOR)[1] # ' question '
    before_answer = sys + SEPERATOR + ques + SEPERATOR+ ' ' # sys ### question ### 


    question_ids = qf_tokenizer(question).input_ids
    text_source_ids = tokenizer(text_source).input_ids
    before_answer_ids = tokenizer(before_answer).input_ids
    mask_len = len(before_answer_ids)
    #target 생성 : text_source 에서 before_answer length을 알아내서 text_source에서 특정 길이만큼 -100채우기
    target = tokenizer(text_source).input_ids
    target[:mask_len] = [-100] * mask_len
    
    result = {
        "input_ids" : text_source_ids, #list
        "qformer_input_ids" : question_ids, #list
        "target" : target, #list
        "im_st" : tokenizer(IMAGE_START).input_ids,
        "im_end" : tokenizer(IMAGE_END).input_ids
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
                 tokenizer: transformers.PreTrainedTokenizer,
                 qf_tokenizer: transformers.PreTrainedTokenizer,
                 data_path: str, 
                 txt_path: str,
                 data_args: DataArguments,
                 is_train: bool):
        super(LazySupervisedDataset_Assembly, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # txt_file = open(txt_path, "r")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.qf_tokenizer = qf_tokenizer
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
 
        # sources = preprocess_multimodal_sequential_reasoning_update(copy.deepcopy(content), self.data_args)

        # input_ids, qformer_input_ids, target, im_st, im_end
        data_dict = preprocess_assembly(
            content,    
            self.tokenizer,
            self.qf_tokenizer,
            has_image=False
        ) #input_ids with chunks


        #load image feature
        single_feature_path = os.path.join(self.feature_path, serial_number+'.npy')
        feat = torch.tensor(np.load(single_feature_path)) #[num_img, 576]
        num_img = feat.shape[0]
        img_getitem = {
            "feat" : feat, #tensor
            "num_img" : num_img, #int
            
        }

        data_dict.update(img_getitem)

        return data_dict

def assembly_collate_fn(batch):
    #text
    b_input_ids = []
    b_qformer_input_ids = []
    b_target = []
    #image
    b_feat = []
    b_feat_mask = []
    b_img_nums = []
    #get largest image num in batch
    for each_batch in batch:
        b_img_nums.append(each_batch["num_img"])
        b_input_ids.append(each_batch["input_ids"])
        b_qformer_input_ids.append(each_batch["qformer_input_ids"])
        b_target.append(each_batch["target"])
    b_max_img = max(b_img_nums)
    im_start_ids = batch[0]["im_st"]
    im_end_ids = batch[0]["im_end"]
    _, num_token, D = batch[0]["feat"].shape 

    #pad text first
    b_p_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids,
                                                    batch_first=True,
                                                    padding_value = self.tokenizer.pad_token_id) #NOTE 이거 뭐임?
    b_p_qformer_input_ids = torch.nn.utils.rnn.pad_sequence(b_qformer_input_ids,
                                                    batch_first=True,
                                                    padding_value = self.qf_tokenizer.pad_token_id) #NOTE 얘는 0이어야할텐데
    b_p_target_ids = torch.nn.utils.rnn.pad_sequence(b_qformer_input_ids,
                                                    batch_first=True,
                                                    padding_value = self.tokenizer.pad_token_id)    
    #llm 넣기 전에 pad attention 을 concat해야 하므로 생성
    text_pad_mask = torch.ne(b_p_input_ids, self.tokenizer.pad_token_id)

    #pad image feats
    for each_batch in batch:
        if each_batch["feat"].shape[0] == b_max_img:
            b_feat.append(each_batch["feat"])
            att_mask = torch.ones((b_max_img, num_token))
            b_feat_mask.append(att_mask)
        else:
            #부족한 만큼 padding 
            each_feat = each_batch["feat"] #[num_img, num_token, D]
            att_mask = torch.ones((each_feat.shape[0], num_token)) #[num_img, num_token]
            marginal = torch.zeros((b_max_img - each_feat.shape[0], num_token, D))
            att_mask_zero = torch.zeros((b_max_img - each_feat.shape[0], num_token))
            each_feat = torch.cat([each_feat, marginal], dim=0)
            att_mask = torch.cat([att_mask, att_mask_zero], dim=0)
            b_feat.append(each_feat)
            b_feat_mask.append(att_mask)
    
    b_feat = torch.stack(b_feat) #[B, max_img, num_token, D]
    b_feat_mask = torch.stack(b_feat_mask, dim=0) #[B, max_img, num_token]이 되도록

    result = {
        "input_ids" : b_p_input_ids,
        "qformer_ids" : b_p_qformer_input_ids,
        "target_ids" : b_p_target_ids,
        "input_ids_pad_mask" : text_pad_mask,

        "images" : b_feat,
        "images_att" : b_feat_mask,
        "image_num_in_batch" : b_img_nums

    }

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    qf_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #text
        b_input_ids = []
        b_qformer_input_ids = []
        b_target = []
        #image
        b_feat = []
        b_feat_mask = []
        b_img_nums = []
        #get largest image num in batch
        for each_batch in instances:
            b_img_nums.append(each_batch["num_img"])
            b_input_ids.append(torch.tensor(each_batch["input_ids"], dtype=torch.long))
            b_qformer_input_ids.append(torch.tensor(each_batch["qformer_input_ids"], dtype=torch.long))
            b_target.append(torch.tensor(each_batch["target"], dtype=torch.long))
        b_max_img = max(b_img_nums)

        im_start_ids = torch.tensor(instances[0]["im_st"], dtype=torch.long)
        im_end_ids = torch.tensor(instances[0]["im_end"], dtype=torch.long)
        _, num_token, D = instances[0]["feat"].shape 
        #pad text first
        b_p_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids,
                                                        batch_first=True,
                                                        padding_value = self.tokenizer.pad_token_id) #NOTE 이거 뭐임?
        b_p_qformer_input_ids = torch.nn.utils.rnn.pad_sequence(b_qformer_input_ids,
                                                        batch_first=True,
                                                        padding_value = self.qf_tokenizer.pad_token_id) #NOTE 얘는 0이어야할텐데
        b_p_target_ids = torch.nn.utils.rnn.pad_sequence(b_target,
                                                        batch_first=True,
                                                        padding_value = -100)    
        #llm 넣기 전에 pad attention 을 concat해야 하므로 생성
        text_pad_mask = torch.ne(b_p_input_ids, self.tokenizer.pad_token_id)

        #pad image feats
        for each_batch in instances:
            if each_batch["feat"].shape[0] == b_max_img:
                b_feat.append(each_batch["feat"])
                att_mask = torch.ones((b_max_img, num_token))
                b_feat_mask.append(att_mask)
            else:
                #부족한 만큼 padding 
                each_feat = each_batch["feat"] #[num_img, num_token, D]
                att_mask = torch.ones((each_feat.shape[0], num_token)) #[num_img, num_token]
                marginal = torch.zeros((b_max_img - each_feat.shape[0], num_token, D))
                att_mask_zero = torch.zeros((b_max_img - each_feat.shape[0], num_token))
                each_feat = torch.cat([each_feat, marginal], dim=0)
                att_mask = torch.cat([att_mask, att_mask_zero], dim=0)
                b_feat.append(each_feat)
                b_feat_mask.append(att_mask)
        
        b_feat = torch.stack(b_feat) #[B, max_img, num_token, D]
        b_feat_mask = torch.stack(b_feat_mask, dim=0) #[B, max_img, num_token]이 되도록

        result = {
            "input_ids" : b_p_input_ids,
            "qformer_ids" : b_p_qformer_input_ids,
            "target_ids" : b_p_target_ids,
            "input_ids_pad_mask" : text_pad_mask,
            "im_start_ids" : im_start_ids,
            "im_end_ids" : im_end_ids,

            "images" : b_feat,
            "images_att" : b_feat_mask,
            "image_num_in_batch" : b_img_nums

        }

        # images = [instance['images'] for instance in instances]
        # batch['images'] = images 
        return result


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                qf_tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset_Assembly(tokenizer=tokenizer,
                                qf_tokenizer= qf_tokenizer,
                                data_path=data_args.data_path,
                                txt_path=data_args.train_txt_path,
                                data_args=data_args, is_train=True)
    val_dataset =  LazySupervisedDataset_Assembly(tokenizer=tokenizer,
                                qf_tokenizer= qf_tokenizer,
                                data_path=data_args.data_path,
                                txt_path=data_args.val_txt_path,
                                data_args=data_args, is_train=False)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, qf_tokenizer=qf_tokenizer)
    
    
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
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4, 
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # 사용 모델
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        llm = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

        print(f"llm device : {llm.device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # from transformers import Blip2QFormerConfig, Blip2QFormerModel
        # configuration_qformer = Blip2QFormerConfig()
        # qformer = Blip2QFormerModel(configuration_qformer)
        
        print("initializing")
        #main model initialize
        model = SequentialMM_Model(llm=llm, query_num=model_args.query_num, args=model_args, device=training_args.device).to(training_args.device)
        print("model finished")
        model.load_mm_projector_state_dict()
        print("a")
        qf_tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_qformer_tokenizer_path)
        # qf_tokenizer = model.init_tokenizer()
        print("b")
        tokenizer.add_tokens(['###', '<im_st>', '<im_end>'], special_tokens=True)
        model.llm.resize_token_embeddings(len(tokenizer))
        training_args.gradient_checkpointing=False
        print("enabling")
        model.llm.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        print("finished")

        #freeze qformer, mm_projector
        #model.qformer.requires_grad_(False)
        #model.mm_projector.requires_grad_(False)



    llm.config.use_cache = False

    if model_args.freeze_backbone:
        llm.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        llm.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    ## PEFT(Parameter-Efficient Fine Tuning)   
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(llm),
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
    
    '''
    Prompt
    '''
    # if 'mpt' in model_args.model_name_or_path: # mpt : Pretrained decoder-only transformer
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path, # 
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right"
    #     )
    # else:
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right",
    #         use_fast=False,
    #     )

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
        Prompt 처리
        '''
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    '''
    VM
    '''
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp # Fully Shared Data Parallel
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        '''
        Projection Layer W
        '''
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False) # VM/LM(?)-> Frozen
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True #  MLP 2 Layer -> Trainable 

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter: # VM Frozen
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False # Frozen

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in llm.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    
    # #main model initialize
    # model = SequentialMM_Model(llm=llm, query_num=model_args.query_num, args=model_args)
    # model.load_mm_projector_state_dict()

    # qf_tokenizer = model.init_tokenizer()
    # tokenizer.add_tokens(['###', '<im_st>', '<im_end>'], special_tokens=True)
    # model.llm.resize_token_embeddings(len(tokenizer))
    # training_args.gradient_checkpointing=False
    # model.llm.gradient_checkpointing_enable()
    
    # breakpoint()

    print(f"lora params : {model.llm.print_trainable_parameters()}")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params : {count_parameters(model)}")



    ## DataLoader
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              qf_tokenizer=qf_tokenizer,
                                              data_args=data_args)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
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