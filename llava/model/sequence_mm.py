import torch
import numpy as np
import sys
sys.path.append("/SeqMMLearning")

from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, PretrainedConfig, LlamaForCausalLM, LlamaConfig
from mamba_ssm import Mamba
from peft import LoraConfig, get_peft_model

from llava.utils import find_all_linear_names
from llava.model.multimodal_projector.builder import build_vision_projector
from typing import Dict, Optional, Sequence, List

class SeqMMConfig(LlamaConfig):
    model_type='seqMM'

    def __init__(self, 
                    model_name_or_path='/SeqMMLearning/checkpoints/llava-v1.5-7b',
                    cache_dir=None, 
                    model_max_length=2048,
                    query_num=32,
                    use_pretrained_qformer=True, 
                    pretrained_qformer_path='/data/pretrained_models/qformer_pretrained',
                    pretrained_qformer_query_token_path='/data/pretrained_models/qformer_pretrained/query_tokens/query_tokens.pth',
                    mm_projector_model_path='/SeqMMLearning/checkpoints/llava-v1.5-7b/mm_projector.bin',
                    **kwargs
                    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.model_max_length = model_max_length
        self.query_num = query_num
        self.use_pretrained_qformer = use_pretrained_qformer
        self.pretrained_qformer_path = pretrained_qformer_path
        self.pretrained_qformer_query_token_path = pretrained_qformer_query_token_path
        self.mm_projector_model_path = mm_projector_model_path


class SequentialMM_Model(PreTrainedModel):
    config_class = SeqMMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #========================llm init and get peft model ============================ NOTE : pretrained된거 load할때는?
        self.llm = LlamaForCausalLM.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
        )
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm.enable_input_require_grads()
        self.llm.config.use_cache = False

        #=========================LLM PEFT ==============================================
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=find_all_linear_names(self.llm),
            lora_dropout=0.05,
            bias='none',
            task_type="CAUSAL_LM",
        )
        self.llm.to(torch.bfloat16)
        print("Adding LoRA adapters...")
        self.llm = get_peft_model(self.llm, lora_config) # LlavaLlamaForCausalLM -> PeftModelForCausalLM 모델 변경
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        self.llm_tokenizer.padding_side="right"
        self.llm_tokenizer.truncation_side="right"

        #=================================mamba init =======================================================
        self.mamba_patch = Mamba(
            d_model=768,
            d_state=4,
            d_conv=4,
            expand=2,
        )

        self.mamba_img_seq = Mamba(
            d_model=768,
            d_state=4,
            d_conv=2,
            expand=2,
        )
        #====================================etc ====================================
        self.ln_vision = nn.Linear(1024, 768)
        print("init mamba finished")
        self.args = args
        self.bridge_former_to_projector = nn.Linear(768, 1024)
        print("build vision projector")
        mm_projector = build_vision_projector(self.llm.config)
        self.mm_projector = mm_projector
        self.device=device

        load_mm_projector_state_dict()
    
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def load_mm_projector_state_dict(self):
        mm_projector_state_dict = torch.load(self.args.mm_projector_model_path)
        new_state_dict={}
        for key, value in mm_projector_state_dict.items():
            if key.startswith("model.mm_projector."):
                new_key = key[len("model.mm.projector."):]
                new_state_dict[new_key] = value
        self.mm_projector.load_state_dict(new_state_dict)


    def forward(self, return_loss=True, **sample):
        image_embeds = sample["image_feat"].to(self.device) # list[ [num_img, 576], [num_img, 576], [num_img, 576] ...] -> len : B / list이므로 gpu로 못올림 
        img_nums = sample["image_num_in_batch"]
        image_embeds = self.ln_vision(image_embeds) # [B, max_img, token, 768]

        B, max_img, token_num, D = image_embeds.shape 
        image_embeds = image_embeds.reshape(B*max_img, token_num, D)



        mamba_patch_embed = self.mamba_patch(image_embeds)[:,-1, :] #[B*max_img, D]

        mamba_patch_embed = mamba_patch_embed.reshape(B, max_img, D)

        #mamba forward
        mamba_embed = self.mamba_img_seq(mamba_patch_embed) #[B, max_img, D]

        #마지막 값들만 배치 단위로 가져와야함
        batch_indices = torch.arange(B)
        adjusted_lengths = torch.tensor(img_nums) - 1
        mamba_final_states = mamba_embed[batch_indices, adjusted_lengths] #[B, D]
        mamba_final_states = self.mm_projector(self.bridge_former_to_projector(mamba_final_states)) #[B,4096]

        mamba_final_states = mamba_final_states.unsqueeze(1) #[B, 1, 4096]
        mamba_atts = torch.ones(mamba_final_states.size()[:-1], dtype=torch.long).to(self.device)

        #이제 text input 만들기
        text_input_tokens = self.llm_tokenizer(
            sample["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in sample['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill( 
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(mamba_atts.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm.get_input_embeddings()(llm_tokens['input_ids']) #NOTE 이게 어떻게 batch단위로 되는지 확인하기 
        inputs_embeds = torch.cat([mamba_final_states, inputs_embeds], dim=1)
        attention_mask = torch.cat([mamba_atts, llm_tokens['attention_mask']], dim=1)

        outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        return outputs



    @torch.no_grad()
    def predict(self, **sample):

        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]
        input_ids_len = sample["input_ids_len"]

        #text input
        text_input = sample["qformer_ids"] #input ids [B, max_Seq]
        text_atts = torch.ne(text_input, 0)
        # text_atts = sample["text_att"] #[B, maxSeq]
        #image embedding
        # image_embeds = self.ln_vision(images_feat) 
        B, max_img, num_token, D = image_embeds.shape #D :768
        _, max_seq = text_input.shape

        #image / text input, attention mask expand
        image_embeds = image_embeds.reshape(B*max_img, num_token, D) #[B*max_img, num_token, D]
        # if not self.args.use_pretrained_qformer:
        image_embeds = self.ln_vision(image_embeds)
        images_atts = images_atts.reshape(B*max_img, -1) #[B*max_img, num_token]
        text_input = text_input.repeat_interleave(max_img,dim=0) #[B*max_img, maxSeq] #NOTE 이거 [1,2,3]이면 [1,1,1, 2,2,2, 3,3,3] 이런식으로 반복해야하는데 맞게 들어가는지 확인해야함
        text_atts = text_atts.repeat_interleave(max_img,dim=0) #NOTE 이것도 확인해야함

        #query token / mask expand, 
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [B*max_img, querynum, D]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device) #[B*max_img, query_num] #NOTE 여기도 마스킹 해줘야함!!!
        #NOTE images_atts에서 maxSeq을 query_num만큼 잘라서 가져오면 될거같음
        query_atts = images_atts[:,:self.num_query_token]

        #get final attention mask
        #qformer attention mask
        qformer_atts = torch.cat([query_atts, text_atts], dim=1) #[B*max_img, query_num + maxSeq]


        query_output = self.qformer(
            text_input, #[B*max_img, maxSeq]
            attention_mask=qformer_atts, #[B*max_img, query_num + maxSeq]
            query_embeds=query_tokens, #[B*max_img, querynum, D]
            encoder_hidden_states=image_embeds, #[B*max_img, num_token, 1024]
            encoder_attention_mask= images_atts, #[B*max_img, num_token]
            return_dict=True,
        ).last_hidden_state[:,:query_tokens.size(1), :]

        #query_output.last_hidden_state[:,:query_tokens.size(1),:] #NOTE 확인한번하기 

        #NOTE [B, max_img*query_token, 768] , mask도 동일하게 처리 
        query_output = query_output.reshape(B, max_img*self.num_query_token, -1) #[B, max_img*self.num_query_token, D]
        query_atts = query_atts.reshape(B, max_img*self.num_query_token) #[B, max_img*self.num_query_token]

        qformer_output = self.mm_projector(self.bridge_former_to_projector(query_output)) #[B, max_img*self.num_query_token, 4096]




        #input_text_ids embedding
        text_input_ids = sample["input_ids"] #[B, maxS] -> [B*maxS] -> [B, maxS, 4096]
        B, maxS = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(B*maxS)
        text_token_embeds = self.llm.get_input_embeddings()(text_input_ids) #[B*maxS, 4096]
        text_token_embeds = text_token_embeds.reshape(B, maxS, -1) #[B, maxS, 4096] 

        target = sample["target_ids"]
        llm_input_rep, llm_input_pad_mask, llm_target = self.process_input_llm(qformer_output, text_token_embeds, target, img_nums, input_ids_len)


        result = self.llm.generate( #110M
            attention_mask=llm_input_pad_mask,
            inputs_embeds=llm_input_rep,
            return_dict=True
        )

        return result

