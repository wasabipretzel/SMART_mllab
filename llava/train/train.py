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
from transformers import AutoTokenizer


from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.sequence_mm import SequentialMM_Model, only_LLM
from llava.config.hf_config import * #argument class
from llava.utils import rank0_print, maybe_zero_3, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_mm_adapter_state_maybe_zero_3, find_all_linear_names, safe_save_model_for_hf_trainer
from llava.dataset.assembly_dataset import *

local_rank = None


def make_supervised_data_module( data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset_Assembly(
                                data_path=data_args.data_path,
                                txt_path=data_args.train_txt_path,
                                data_args=data_args, is_train=True)
    val_dataset =  LazySupervisedDataset_Assembly(
                                data_path=data_args.data_path,
                                txt_path=data_args.val_txt_path,
                                data_args=data_args, is_train=False)

    data_collator = DataCollatorForSupervisedDataset()
    
    
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower == 'None':
        model_args.vision_tower=None

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    config_seqmm = SeqMMConfig(model_args.model_name_or_path, 
        training_args.cache_dir, 
        training_args.model_max_length,
        model_args.query_num,
        model_args.use_pretrained_qformer,
        model_args.pretrained_qformer_path, 
        model_args.pretrained_qformer_query_token_path,
        model_args.mm_projector_model_path)

    
    print("initializing")
    #main model initialize
    model = SequentialMM_Model(config_seqmm).to(training_args.device)
    training_args.gradient_checkpointing=False


    '''
    Prompt 처리
    '''
    model.llm_tokenizer.pad_token = model.llm_tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

  

    print(f"lora params : {model.llm.print_trainable_parameters()}")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params : {count_parameters(model)}")

    ## DataLoader
    data_module = make_supervised_data_module(data_args=data_args)

    trainer = LLaVATrainer(model=model,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("tldr-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.llm.config.save_pretrained(training_args.output_dir)
            model.llm.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()