import torch
import numpy as np
import sys
sys.path.append("/SeqMMLearning")

from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig
from typing import Dict, Optional, Sequence, List
from transformers import VivitModel


class SequentialMM_Model(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vid_enc = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.cls = nn.Linear(768, 91)

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



