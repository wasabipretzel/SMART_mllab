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

from dataset.smart import *
from config.hf_config import *
from models.basemodel import *
from models.build_model import get_model
from metrics.build_metric import get_metric
from dataset.build_dataset import get_dataset
from utils.util import count_parameters, NoWarningFilter


local_rank = None
logger = logging.get_logger("transformers")

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    if training_args.report_to == ['wandb']:
        os.environ["WANDB_PROJECT"] = training_args.project_name

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info("initializing")

    #using get_model 
    model, processor = get_model(model_args, training_args)

    logger.info(f"Trainable model params : {count_parameters(model)}")

    # craete dataset & collator
    data_module = get_dataset(model_args, data_args, processor=processor)

    metric = get_metric(model_args, processor)

    trainer = Seq2SeqTrainer(model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    tokenizer=processor.tokenizer if model_args.model_type=="instructblip" else None, #for prediction
                    **data_module)


    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    train()