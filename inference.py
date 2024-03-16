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

    logging.info("Loading Pretrained model")
    #load model 
    SeqMM_config = PretrainedConfig.from_pretrained(training_args.load_ckpt_path)
    model = SequentialMM_Model.from_pretrained(pretrained_model_name_or_path=training_args.load_ckpt_path,
                                                config= SeqMM_config
                                                ).to(training_args.device)
  

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"total params : {count_parameters(model)}")

    ## DataLoader
    data_module = make_test_module(data_args=data_args)

    trainer = Trainer(model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    data_collator = data_module["data_collator"]
                    )


    predictions, labels, metrics = trainer.predict(test_dataset=data_module["test_dataset"])
    logging.info(metrics)

    #TODO need to add prediction / labels saving module (Optional)

if __name__ == "__main__":
    inference()