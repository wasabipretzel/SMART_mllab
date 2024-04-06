import os
import sys
# sys.path.append("/SeqMM")
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
from transformers import Trainer, set_seed, Seq2SeqTrainer
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



def make_supervised_data_module(data_args, processor=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SMART(
                        data_args=data_args, mode='train')
    val_dataset =  SMART(
                        data_args=data_args, mode='val')

    data_collator = SMART_collator(processor=processor)
    
    
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
    if training_args.report_to == ['wandb']:
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

    #using get_model 
    model, processor = get_model(model_args, training_args)
    # breakpoint()
    #NOTE : if you want to get pretrained model from url, run below code
    # model.VLM.config.text_config.pad_token_id=2  #https://github.com/salesforce/LAVIS/issues/328 , https://github.com/huggingface/transformers/issues/22546#issuecomment-1561257076


    logger.info(f"Trainable model params : {count_parameters(model)}")

    # craete dataset & collator
    data_module = make_supervised_data_module(data_args=data_args, processor=processor)
    metric = ComputeMetric(tokenizer=processor.tokenizer)

    trainer = Seq2SeqTrainer(model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    tokenizer=processor.tokenizer, #for prediction
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