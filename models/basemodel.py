import sys

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig, GenerationConfig
from transformers.utils import logging
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model
from models.instructblip.configuration_instructblip import InstructBlipConfig
from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration
from utils.util import NoWarningFilter

logger = logging.get_logger('transformers')
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating (max_new_token), add custom filter

# class BaseModel(PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         if self.config.train_mode == True:
#             self.VLM = InstructBlipForConditionalGeneration.from_pretrained(config.pretrained_model_path, config.image_size, config.s2wrapper, ignore_mismatched_sizes=True) # add s2wrapper
#         else:
#             model_name = self.config.pretrained_model_path.split("/")[-1]
#             self.vlm_config=InstructBlipConfig.from_pretrained(f"/SMART_mllab/models/instructblip/{model_name}.json")
#             self.VLM = InstructBlipForConditionalGeneration(self.vlm_config)
#         if config.model_type=="instructblip_vicuna":
#             self.VLM.language_model.generation_config.eos_token_id=2
#             self.VLM.generation_config.eos_token_id=2
#             logger.info(f"Generation config modified same as training config : {self.VLM.generation_config}")
#         self.generation_config = self.VLM.generation_config
#         if config.freeze_llm:
#             for param in self.VLM.language_model.parameters(): 
#                 param.requires_grad=False 
#         if config.freeze_image_encoder: #900M
#             for param in self.VLM.vision_model.parameters():
#                 param.requires_grad=False  
#         if config.use_bf16:
#             self.VLM.to(torch.bfloat16)
#         if config.use_lora:
#             logger.info("Adding LoRA adapters...")
#             if config.model_type=="instructblip_vicuna":
#                 lora_config = LoraConfig(
#                     r=config.lora_r,
#                     lora_alpha=config.lora_alpha,
#                     target_modules=["q_proj","v_proj"],
#                     lora_dropout=config.lora_dropout,
#                     bias='none',
#                     task_type="CAUSAL_LM",
#                 )
#             elif config.model_type=="instructblip_flant5":
#                 lora_config = LoraConfig(
#                     r = config.lora_r,
#                     lora_alpha=config.lora_alpha,
#                     target_modules=["q", "v"],
#                     lora_dropout=config.lora_dropout,
#                     bias="none",
#                     task_type="SEQ_2_SEQ_LM"
#                 )
#             else:
#                 raise NotImplementedError

#             self.VLM.language_model = get_peft_model(self.VLM.language_model, lora_config)

#     def forward(self, return_loss=True, **sample):
#         result = self.VLM(
#             **sample
#         ) #['loss', 'logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']

#         #language_model_outputs.logits.shape -> [B, S, vocab]
#         # logits은 language_model_outputs.logits과 동일함 
        
#         output = {
#             "loss" : result.loss,
#             "logits" : result.logits,
#             "keyval_class" : result.keyval_pred,
#         }
        
#         return output

#     @torch.no_grad()
#     def generate(self, **sample):

#         result = self.VLM.generate(
#             **sample,
#             #generation_kwargs : if no parameters, -> greedy search
#             # max_length = self.generation_config.max_length
#         )

#         return result




class BaseModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.train_mode == True:
            self.VLM = InstructBlipForConditionalGeneration.from_pretrained(config.pretrained_model_path)
        else:
            model_name = self.config.pretrained_model_path.split("/")[-1]
            self.vlm_config=InstructBlipConfig.from_pretrained(f"/SMART_mllab/models/instructblip/{model_name}.json")
            self.VLM = InstructBlipForConditionalGeneration(self.vlm_config)
        if config.model_type=="instructblip_vicuna":
            self.VLM.language_model.generation_config.eos_token_id=2
            self.VLM.generation_config.eos_token_id=2
            logger.info(f"Generation config modified same as training config : {self.VLM.generation_config}")
        self.generation_config = self.VLM.generation_config
        if config.freeze_llm:
            for param in self.VLM.language_model.parameters(): 
                param.requires_grad=False 
        if config.freeze_image_encoder: #900M
            for param in self.VLM.vision_model.parameters():
                param.requires_grad=False  
        if config.use_bf16:
            self.VLM.to(torch.bfloat16)
        if config.use_lora:
            logger.info("Adding LoRA adapters...")
            if config.model_type=="instructblip_vicuna":
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q_proj","v_proj"],
                    lora_dropout=config.lora_dropout,
                    bias='none',
                    task_type="CAUSAL_LM",
                )
            elif config.model_type=="instructblip_flant5":
                lora_config = LoraConfig(
                    r = config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q", "v"],
                    lora_dropout=config.lora_dropout,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM"
                )
            else:
                raise NotImplementedError

            self.VLM.language_model = get_peft_model(self.VLM.language_model, lora_config)
    
        # self.config.use_SAM=False
        # self.config.use_dynamic_sam=False
        # self.config.white_image_crossattention=False
        # self.config.use_onlySAM=False
        # Linear map for SAM feature 
        if self.config.use_SAM or self.config.use_dynamic_sam:
            self.sam_linear = nn.Linear(self.config.sam_feat_dim, self.VLM.config.vision_config.hidden_size) #1280, 1408
            if config.use_bf16:
                self.sam_linear.to(torch.bfloat16)
        


    def forward(self, return_loss=True, **sample):
        if self.config.use_SAM or self.config.use_dynamic_sam:
            if self.config.use_bf16:
                sample["sam_feature"] = sample["sam_feature"].to(torch.bfloat16)
            sample["sam_feature"] = self.sam_linear(sample["sam_feature"]) #[B, 256, 1408]
        if self.config.white_image_crossattention == False:
            sample["white_image_crossattention"] = False
        if self.config.use_onlySAM:
            sample["use_onlySAM"] = True
        result = self.VLM(
            **sample
        ) #['loss', 'logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']
        
        output = {
            "loss" : result.loss,
            "logits" : result.logits,
            "keyval_class" : [result.keyval_pred],
        }
        
        return output

    @torch.no_grad()
    def generate(self, **sample):
        if self.config.use_SAM or self.config.use_dynamic_sam:
            if self.config.use_bf16:
                sample["sam_feature"] = sample["sam_feature"].to(torch.bfloat16)
            sample["sam_feature"] = self.sam_linear(sample["sam_feature"]) #[B, 256, 1408]
        if self.config.white_image_crossattention == False:
            sample["white_image_crossattention"] = False
        if self.config.use_onlySAM:
            sample["use_onlySAM"] = True
        result = self.VLM.generate(
            **sample,
            #generation_kwargs : if no parameters, -> greedy search
            # max_new_tokens=2
            # max_length = self.generation_config.max_length
        )

        return result


