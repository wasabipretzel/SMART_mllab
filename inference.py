import os
import sys
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
from transformers import Trainer, Seq2SeqTrainer
from datasets import load_metric

from dataset.smart import *
from config.hf_config import *
from models.basemodel import *
from models.build_model import get_model
from utils.util import count_parameters


local_rank = None
logger = logging.getLogger(__name__)


@dataclass
class ComputeMetric:
    tokenizer: transformers.PreTrainedTokenizer
    """
    EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids)
        get all logits, labels after all eval_step
       pred.predictions (얘가 맞는듯) (300, 182)
       pred.label_ids  (얜 죄다 -100) (300, 124)
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
       tokenizer을 넣어줘야하는듯. 
    """
    metric = load_metric("accuracy")
    candidates = {
        "A" : 0,
        "B" : 1,
        "C" : 2,
        "D" : 3,
        "E" : 4,
    }
    def compute_metrics(self, pred):

        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id #fill -100 index with pad_token_id (preventing index/overflow error)
        gt_answer_list = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True) #get rid of pad tokens

        #prediction 
        pred.predictions[pred.predictions == -100] = self.tokenizer.pad_token_id
        pred_answer_list = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        breakpoint()
        gt_filtered = []
        pred_filtered = []
        for gt, pred_ans in zip(gt_answer_list, pred_answer_list):
            gt_flag=False
            pred_ans_flag=False
            for each_option in self.candidates.keys():
                if each_option in gt and gt_flag == False:
                    gt_filtered.append(self.candidates[each_option])
                    gt_flag=True
                if each_option in pred_ans and pred_ans_flag == False:
                    pred_filtered.append(self.candidates[each_option])
                    pred_ans_flag=True 
            # pred에 아예 A,B,C,D,E 없는 경우
            if pred_ans_flag == False:
                pred_filtered.append(-1)
        
        metrics = self.metric.compute(references=gt_filtered, predictions=pred_filtered)

        return metrics


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

    logger.info("Loading Pretrained model")
    #load model 
    model, processor = get_model(model_args, training_args)

    ## DataLoader
    data_module = make_test_module(data_args=data_args, processor=processor)
    metric = ComputeMetric(processor.tokenizer)

    trainer = Seq2SeqTrainer(model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    data_collator = data_module["data_collator"]
                    )


    predictions, labels, metrics = trainer.predict(test_dataset=data_module["test_dataset"])
    logger.info(metrics)

    #TODO need to add prediction / labels saving module (Optional)

if __name__ == "__main__":
    inference()