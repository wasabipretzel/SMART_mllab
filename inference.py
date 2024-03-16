import os
import sys
sys.path.append("/SeqMMLearning")
import copy
import json
import logging
import pathlib
import re
import requests
from typing import Dict, Optional, Sequence, List
from itertools import accumulate

import torch
import numpy as np
import transformers
from transformers import Trainer

from dataset.dataset_vivit import *
from config.hf_config import *
from models.vivit import *
from utils.util import *


local_rank = None
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    """
        get all logits, labels after all eval_step
        
    """
    class_logit = pred.predictions[0] #[total_samples, class] logit
    labels = pred.label_ids #[total_samples, class] multi hot label
    
    pred_score = torch.sigmoid(torch.tensor(class_logit))

    aps = compute_multiple_aps(labels, pred_score)

    metrics = {
        "mAP" : aps[aps != -1].mean()
    }
    return metrics



def make_test_module( data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    test_dataset =  MOMA(data_args=data_args, mode='test')

    data_collator = MOMA_collator()
    
    
    return dict(test_dataset=test_dataset,
                data_collator=data_collator)



def inference():
    """
        test dataset initialize 이후, trainer init시에 train_dataset, eval_dataset 둘 다 None으로 해놓고
        trainer.predict(test_dataset, ~~)
    """

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if training_args.report_to != 'none':
        os.environ["WANDB_PROJECT"] = training_args.project_name
    
    print("initializing")
    #main model initialize
    # model = SequentialMM_Model(model_args).to(training_args.device)

    #load model 
    SeqMM_config = PretrainedConfig.from_pretrained(training_args.load_ckpt_path)
    model = SequentialMM_Model.from_pretrained(pretrained_model_name_or_path=training_args.load_ckpt_path,
                                                config= SeqMM_config
                                                ).to(training_args.device)
  

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params : {count_parameters(model)}")

    ## DataLoader
    data_module = make_test_module(data_args=data_args)

    trainer = Trainer(model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    data_collator = data_module["data_collator"]
                    )


    predictions, labels, metrics = trainer.predict(test_dataset=data_module["test_dataset"])
    # trainer.save_model(training_args.output_dir)
    # trainer.save_state()

    print(metrics)

    # model.config.use_cache = True

    # if model_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )
    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters()
    #     )
    #     if training_args.local_rank == 0 or training_args.local_rank == -1:
    #         model.llm.config.save_pretrained(training_args.output_dir)
    #         model.llm.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    # else:
    #     safe_save_model_for_hf_trainer(trainer=trainer,
    #                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    inference()