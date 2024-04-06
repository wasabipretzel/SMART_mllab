"""
    source file for huggingface Argparser.
    Arguments classes must inherit dataclasses.
"""
from dataclasses import dataclass, field
import transformers
from transformers import TrainingArguments, PretrainedConfig, Seq2SeqTrainingArguments
from typing import Dict, Optional, Sequence, List
from functools import partial


@dataclass
class ModelArguments(PretrainedConfig):
    model_type:str="instructblip"
    pretrained_model_path: str="Salesforce/instructblip-vicuna-7b" #can be switched with url or saved pretrained model path 
    freeze_llm: bool=True
    freeze_image_encoder: bool=True
    use_bf16: bool=True
    # keys_to_ignore_at_inference: List[str]=field(default_factory=partial(list, ["qformer_outputs", 'vision_outputs'])) -> model forward return 의 key값을 바꿔주면 됨
    

@dataclass
class DataArguments:
    split_type: str="PS"
    split_path: str="/data/split"
    data_path: str="/data/SMART101-release-v1/SMART101-Data"
    num_class: int=91


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
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
    ddp_find_unused_parameters: bool=False
    #generation arguments in trainer evaluate()
    predict_with_generate: bool=True # evaluate시 AR방식으로 생성해서 결과 뽑아주게함. False면 teacher forcing
    max_length=256
