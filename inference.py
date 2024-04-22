import os
import sys
import copy
import json
import pathlib
import re
import requests
from typing import Dict, Optional, Sequence, List
from itertools import accumulate

import torch
import numpy as np
import transformers
from transformers import Trainer, set_seed, Seq2SeqTrainer
from transformers.utils import logging
from datasets import load_metric

from config.hf_config import *
from models.build_model import get_model
from metrics.build_metric import get_metric
from dataset.build_dataset import get_dataset
from utils.util import count_parameters, NoWarningFilter

local_rank = None
logger = logging.get_logger("transformers")
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating, add custom filter


def make_test_module(data_args, processor=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = SMART(data_args=data_args, mode='train')
    test_dataset = SMART(data_args=data_args, mode='test')

    data_collator = SMART_collator(processor=processor)
    
    
    return dict(
                test_dataset=test_dataset,
                data_collator=data_collator)


def inference():
    """
        test dataset initialize 이후, trainer init시에 train_dataset, eval_dataset 둘 다 None으로 해놓고
        trainer.predict(test_dataset, ~~)
    """

    global local_rank
    mode = 'test'
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    if training_args.report_to != 'none':
        os.environ["WANDB_PROJECT"] = training_args.project_name
    
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info("Loading Pretrained model")
    #load model 
    model, processor = get_model(model_args, training_args)

    ## DataLoader
    data_module = get_dataset(model_args, data_args, mode, processor=processor)

    embeddings = copy.deepcopy(model.VLM.language_model.get_input_embeddings())
    metric = get_metric(model_args, data_args, processor, embeddings, data_module["eval_dataset"])

    trainer = Seq2SeqTrainer(model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    data_collator = data_module["data_collator"],
                    tokenizer=processor.tokenizer if model_args.model_type=="instructblip" else None,
                    )


    predictions, labels, metrics = trainer.predict(test_dataset=data_module["eval_dataset"])
    logger.info(metrics)

    #TODO need to add prediction / labels saving module (Optional)

if __name__ == "__main__":
    inference()