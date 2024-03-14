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
    vivit_hidden_dim: int =768
    moma_label: int=91
    

@dataclass
class DataArguments:
    split_path: str ="/data/MOMA/preprocessed/splits/split_exclude_over900_and_instover250_nosubact.json"
    raw_vid_path: str="/data/MOMA/raw"
    target_path : str="/data/MOMA/frozen_meta.json"
    sample_frames: int=32
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
