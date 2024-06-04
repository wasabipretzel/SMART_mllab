import sys

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig, GenerationConfig
from transformers.utils import logging
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model

from models.flant5.modeling_t5 import T5ForConditionalGeneration
from utils.util import NoWarningFilter


logger = logging.get_logger('transformers')
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating (max_new_token), add custom filter


class BaseLLMModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.LLM = T5ForConditionalGeneration.from_pretrained(config.llm_pretrained_model_path)
        if config.llm_model_type=="vicuna":
            self.LLM.generation_config.eos_token_id=2
            self.LLM.generation_config.eos_token_id=2
            logger.info(f"Generation config modified same as training config : {self.LLM.generation_config}")
        self.generation_config = self.LLM.generation_config
        if config.freeze_llm:
            for param in self.LLM.parameters(): 
                param.requires_grad=False 
        if config.use_bf16:
            self.LLM.to(torch.bfloat16)
        if config.use_lora:
            logger.info("Adding LoRA adapters...")
            if config.llm_model_type=="vicuna":
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q_proj","v_proj"],
                    lora_dropout=config.lora_dropout,
                    bias='none',
                    task_type="CAUSAL_LM",
                )
            elif config.llm_model_type=="flant5":
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

            self.LLM = get_peft_model(self.LLM, lora_config)
    

    def forward(self, return_loss=True, **sample):
        result = self.LLM(
            **sample
        ) #['loss', 'logits', 'language_model_outputs']
        
        output = {
            "loss" : result.loss,
            "logits" : result.logits,
        }
        
        return output

    @torch.no_grad()
    def generate(self, **sample):
        result = self.LLM.generate(
            **sample,
            #generation_kwargs : if no parameters, -> greedy search
            # max_new_tokens=2
            # max_length = self.generation_config.max_length
        )

        return result