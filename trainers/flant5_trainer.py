"""_summary_
    trainer for training instructblip base model. Specifically, purpose of this trainer is for applying 
    different learning rates for pretrained/scratch modules in model
"""
import numpy as np
from transformers import Seq2SeqTrainer
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.trainer_utils import has_length, EvalLoopOutput, EvalPrediction, PredictionOutput, denumpify_detensorize
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, is_torch_xla_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)

class T5Trainer(Seq2SeqTrainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        In instructblip base model, 
        pretrained module parts are ["qformer", "language_projection"] (need to use very small lr)
        scratch module parts are ["llm lora", "qformer lora"(not implemented), "vit adapter"(not implemented)] (need to use relatively large lr)

        #TODO : pretrained_parameters, scratch_parameters에서 bias은 어떻게 처리해주는게 맞지?
        # t5는 pretrained_lr, scratch_lr 있는 게 맞는지, 아니면 learning_rate만 있는지
        """ 
        opt_model = self.model
        # pretrained_modules = ["qformer", "language_projection"]
        scratch_modules = ["lora"]

        if self.args.pretrained_module_lr is not None and self.args.scratch_module_lr is not None: 
            pretrained_parameters, scratch_parameters= list(), list()
            for name, param in opt_model.named_parameters():
                # for module in pretrained_modules:
                #     if module in name and param.requires_grad:
                #         pretrained_parameters.append(param)
                for module in scratch_modules:
                    if module in name and param.requires_grad:
                        scratch_parameters.append(param)

            optimizer_grouped_parameters = [
                # {'params': pretrained_parameters, 'lr': self.args.pretrained_module_lr},
                {'params': scratch_parameters, 'lr': self.args.scratch_module_lr}
            ]

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        return self.optimizer


    # def log(self, logs:Dict[str, float]):
    #     """
    #         logs : {'loss': 4.2812, 'learning_rate': 7.042253521126761e-11}
    #         -> learning_rate을 빼고 pretrained_lr, scratch_lr로 보고하도록..
    #     """
    #     if 'eval' in list(logs.keys())[0]:
    #         super().log(logs)
    #     else:
    #         # key_names = ["pretrained_lr", "scratch_lr"]
    #         key_names = ["scratch_lr"]
    #         step = logs.get("step", 0)
    #         if "learning_rate" in logs.keys():
    #             del logs["learning_rate"]
    #         lr_dict = {
    #             f'{key_names[i]}': group['lr'] for i, group in enumerate(self.optimizer.param_groups)
    #         }
    #         logs.update(lr_dict)
    #         super().log(logs)