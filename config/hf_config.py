"""
    source file for huggingface Argparser.
    Arguments classes must inherit dataclasses.
"""
from dataclasses import dataclass, field
import transformers
from transformers import TrainingArguments, PretrainedConfig
from typing import Dict, Optional, Sequence, List
from functools import partial


@dataclass
class ModelArguments(PretrainedConfig):
    model_type:str="instructblip"
    vivit_hidden_dim: int =768
    moma_label: int=91
    

@dataclass
class DataArguments:
    split_type: str="PS"
    split_path: str="/data/split"
    data_path: str="/data/SMART101-release-v1/SMART101-Data"
    num_class: int=91


@dataclass
class TrainingArguments(TrainingArguments):
    """
        trainingarguments을 상속받았기 때문에 num_train_epochs, per_device_train_batch_size등이 자동으로 들어감 
    """
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    logging_steps: int=1
    project_name: str=field(default="huggingface")
    label_names: List[str]=field(default_factory=partial(list, ["labels"]))
    load_ckpt_path: str=field(default=None)
    seed: int=42
    should_log: bool=True
