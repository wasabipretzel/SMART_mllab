import os
import sys
sys.path.append("/SeqMM")
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
from transformers import Trainer, set_seed

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



def make_supervised_data_module( data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MOMA(
                        data_args=data_args, mode='train')
    val_dataset =  MOMA(
                        data_args=data_args, mode='val')

    data_collator = MOMA_collator()
    
    
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    if training_args.report_to == 'wandb':
        os.environ["WANDB_PROJECT"] = training_args.project_name

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info("initializing")
    #main model initialize
    #TODO : should PR for not to log config when loading pretraining models.. 
    model = SequentialMM_Model(model_args).to(training_args.device)
  
    #TODO : move this to utils
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"total params : {count_parameters(model)}")

    # craete dataset & collator
    data_module = make_supervised_data_module(data_args=data_args)

    trainer = Trainer(model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    **data_module)


    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    #NOTE should be added when applying lora -> added from llava github
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
    train()