import sys

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig, GenerationConfig
from transformers.utils import logging
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model

from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration
from utils.util import NoWarningFilter

logger = logging.get_logger('transformers')
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating (max_new_token), add custom filter

class BaseModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.VLM = InstructBlipForConditionalGeneration.from_pretrained(config.pretrained_model_path)
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
    

        # Linear map for SAM feature 
        if self.config.use_SAM:
            self.sam_linear = nn.Linear(self.config.sam_feat_dim, self.VLM.config.vision_config.hidden_size) #1280, 1408
            if config.use_bf16:
                self.sam_linear.to(torch.bfloat16)
        


    def forward(self, return_loss=True, **sample):
        if self.config.use_SAM:
            if self.config.use_bf16:
                sample["sam_feature"] = sample["sam_feature"].to(torch.bfloat16)
            sample["sam_feature"] = self.sam_linear(sample["sam_feature"]) #[B, 256, 1408]
        if self.config.white_image_crossattention == False:
            sample["white_image_crossattention"] = False
        if self.config.use_onlySAM:
            sample["use_onlySAM"] = True
        if self.config.llm_only:
            sample["llm_only"] = True
        if self.config.category_classification_loss:
            sample["category_classification_loss"] = True
        result = self.VLM(
            **sample
        ) #['loss', 'logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']
        
        if self.config.category_classification_loss:
            output = {
                "loss" : result.loss,
                "logits" : result.logits,
                "category_loss" : result.category_loss
            }
        else:
            output = {
                "loss" : result.loss,
                "logits" : result.logits,
            }
        
        return output

    @torch.no_grad()
    def generate(self, **sample):
        if self.config.use_SAM:
            if self.config.use_bf16:
                sample["sam_feature"] = sample["sam_feature"].to(torch.bfloat16)
            sample["sam_feature"] = self.sam_linear(sample["sam_feature"]) #[B, 256, 1408]
        if self.config.white_image_crossattention == False:
            sample["white_image_crossattention"] = False
        if self.config.use_onlySAM:
            sample["use_onlySAM"] = True
        if self.config.llm_only:
            sample["llm_only"] = True
        if self.config.category_classification_loss:
            sample["category_classification_loss"] = True
        result = self.VLM.generate(
            **sample,
            #generation_kwargs : if no parameters, -> greedy search
            # max_new_tokens=2
            # max_length = self.generation_config.max_length
        )

        return result




