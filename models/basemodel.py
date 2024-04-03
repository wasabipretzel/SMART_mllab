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
        self.VLM = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")

        self.criterion = nn.MultiLabelSoftMarginLoss()

    def forward(self, return_loss=True, **sample):
        B, T, C, H, W = sample["vid_input"].shape
        
        output = self.vid_enc(sample["vid_input"])

        output = output.pooler_output #[B, 768]

        class_output = self.cls(output) #[B, class]

        result = {
            "loss" : self.criterion(class_output, sample["labels"]),
            "logits" : class_output, #[B, class] logit
            "labels" : sample["labels"],
        }
        
        return result



