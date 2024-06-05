"""
    source file for huggingface Argparser.
    Arguments classes must inherit dataclasses.
"""
import copy

from dataclasses import dataclass, field
import transformers
from transformers import TrainingArguments, PretrainedConfig, Seq2SeqTrainingArguments
from typing import Dict, Optional, Sequence, List, Any
from functools import partial


@dataclass
class ModelArguments(PretrainedConfig):
    model_type: str=field(default=None)#["instructblip_vicuna", "instructblip_flant5", "R50_BERT"]
    pretrained_model_path: str="Salesforce/instructblip-flan-t5-xl"#can be switched with url or saved pretrained model path , "Salesforce/instructblip-flan-t5-xxl"
    freeze_llm: bool=True
    freeze_image_encoder: bool=True
    use_bf16: bool=True
    use_lora: bool=True
    lora_r: int=16
    lora_alpha: int=32
    lora_dropout: float=0.05
    smart_starter_pretrained_path: str=field(default=None) 
    image_size: int=224
    s2wrapper: bool=False
    image_encoder: str=field(default='vit') # ['vit', 'swin', 'vit_384']

@dataclass
class DataArguments:
    split_type: str="PS"
    split_path: str="/home/work/g-earth-22/VLM/VLM/database/SMART-101/data/split"
    data_path: str="/home/work/g-earth-22/VLM/VLM/database/SMART-101/data/SMART101-release-v1/SMART101-Data"
    puzzle_path: str="/home/work/g-earth-22/VLM/VLM/database/SMART-101/data/SMART101-release-v1/puzzle_type_info.csv"
    # num_class: int=91
    prediction_type: str=field(default="answerkey") #could be ["answerkey","answervalue"]. answerkey predict one of 'A','B','C','D','E'. answervalue predict float/str value
    data_image_size: int=224
    add_cot: bool = False
    add_puzzle_option: bool = False
    puzzle_type: str = None
    permutation: bool = False
    permutation_option: str=field(default='opt-shift') # opt-reverse, opt-shift
    add_data: bool = False
    split_eval: bool = False
    split_eval_option : bool = False

    use_caption: bool=False
    category_classification_mapping_path: str=field(default=None)
    sam_feature_path: str=field(default=None)
    SAM_token_mask: bool=False
    
    #for submission
    challenge_phase: str=field(default=None)

    use_dynamic_sam_decoder: bool=False 
    use_dynamic_sam_encoder: bool=False
    use_dynamic_caption: bool=False
    sam_pretrained_model_path: str=field(default=None)
    qwen_pretrained_model_path: str=field(default=None)



@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
        trainingarguments을 상속받았기 때문에 num_train_epochs, per_device_train_batch_size등이 자동으로 들어감 
    """
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    logging_steps: int=1
    project_name: str=field(default="smart_challenge")
    label_names: List[str]=field(default_factory=partial(list, ["labels"]))
    load_ckpt_path: str=field(default=None)
    seed: int=42
    should_log: bool=True
    ddp_find_unused_parameters: bool=False
    pretrained_module_lr: float=field(default=1e-6) #learning rate for pretrained moduel
    scratch_module_lr: float=field(default=1e-4) #learning rate for modules which are trained from scratch
    #generation arguments in trainer evaluate()
    predict_with_generate: bool=True # evaluate시 AR방식으로 생성해서 결과 뽑아주게함. False면 teacher forcing
    max_length: int = 256

    class_ensemble: bool=False
    load_key_ckpt_path: str=field(default=None)
    load_value_ckpt_path: str=field(default=None)

