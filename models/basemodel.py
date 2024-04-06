import sys
# sys.path.append("/SeqMM")

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig
from typing import Dict, Optional, Sequence, List

from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration

class BaseModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.VLM = InstructBlipForConditionalGeneration.from_pretrained("/data/pretrained_models/instructblip-vicuna-7b")
        if config.freeze_llm:
            for param in self.VLM.language_model.parameters():
                param.requires_grad=False 
        if config.freeze_image_encoder:
            for param in self.VLM.vision_model.parameters():
                param.requires_grad=False  
        if config.use_bf16:
            self.VLM.to(torch.bfloat16)

    def forward(self, return_loss=True, **sample):
        result = self.VLM(
            **sample
        ) #['loss', 'logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']

        #language_model_outputs.logits.shape -> [B, S, vocab]

        #logits vs language_mode_outputs?
        # logits은 language_model_outputs.logits과 동일함 
        
        output = {
            "loss" : result.loss,
            "logits" : result.logits
        }
        
        return output

    @torch.no_grad()
    def generate(self, **sample):
        #TODO : 올라오는 sample에 answer text도 있어야함. 그래야 metric 계산이 가능 
        result = self.VLM.generate(
            **sample,
            #generation_kwargs
            do_sample=False,
            num_beams=5,
            max_length=5,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,  
        )

        return result




