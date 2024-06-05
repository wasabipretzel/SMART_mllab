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
from PIL import Image
from transformers import Trainer, set_seed, Seq2SeqTrainer
from transformers.utils import logging
from torch.utils.data import Dataset

from config.hf_config import *
from models.build_model import get_model
from metrics.build_metric import get_metric
from dataset.build_dataset import get_dataset
from utils.util import count_parameters, NoWarningFilter

from trainers.submission_trainer import SubmissionTrainer

local_rank = None
logger = logging.get_logger("transformers")
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating, add custom filter
    
VLAR_CHALLENGE_submission_root = '/submission/'    
def make_response_json(metrics):
    if not os.path.exists(VLAR_CHALLENGE_submission_root):
        os.mkdir(VLAR_CHALLENGE_submission_root)
    
    submission_json = {
        "VLAR" : [

        ]
    }
    
    answer_prediction_list = metrics["test_predicted_answers"]
    answer_prediction_pids = metrics["test_pids"]

    for pid, pred in zip(answer_prediction_pids, answer_prediction_list):
        tmp_sub = {
            "Id" : str(pid),
            "Answer" : str(pred)
        }
        submission_json["VLAR"].append(tmp_sub)

    with open(os.path.join(VLAR_CHALLENGE_submission_root, "submission.json"), "w") as f:
        json.dump(submission_json, f)
    
    print(f"submission.json saved at {os.path.join(VLAR_CHALLENGE_submission_root, 'submission.json')}")
    return


def submit_challenge():
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
    data_args.local_rank = local_rank
    data_args.pretrained_model_path = model_args.pretrained_model_path
    data_args.load_key_ckpt_path = training_args.load_key_ckpt_path
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

    embeddings = copy.deepcopy(model.keymodel.VLM.language_model.get_input_embeddings())
    metric = get_metric(model_args, data_args, processor, embeddings, data_module["eval_dataset"].eval_infos)


    trainer = SubmissionTrainer(model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    data_collator = data_module["data_collator"],
                    tokenizer=processor.tokenizer if "instructblip" in model_args.model_type else None,
    )
    # trainer = Seq2SeqTrainer(model=model,
    #                 args=training_args,
    #                 compute_metrics=metric.compute_metrics,
    #                 data_collator = data_module["data_collator"],
    #                 tokenizer=processor.tokenizer if "instructblip" in model_args.model_type else None,
    #                 )


    predictions, labels, metrics = trainer.predict(test_dataset=data_module["eval_dataset"])
    make_response_json(metrics) # dump the model predicted answers into a json file for evaluation.

    #TODO need to add prediction / labels saving module (Optional)

if __name__ == "__main__":
    submit_challenge()