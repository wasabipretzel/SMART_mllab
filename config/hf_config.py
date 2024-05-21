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
    pretrained_model_path: str="Salesforce/instructblip-vicuna-7b"#can be switched with url or saved pretrained model path , "Salesforce/instructblip-flan-t5-xxl"
    freeze_llm: bool=True
    freeze_image_encoder: bool=True
    use_bf16: bool=True
    use_lora: bool=True
    lora_r: int=16
    lora_alpha: int=32
    lora_dropout: float=0.05
    smart_starter_pretrained_path: str=field(default=None) 
    
    #for SAM feature experiment
    use_SAM: bool=False
    sam_feat_dim: int=1280
    use_onlySAM: bool=False #vit feature을 쓰는 것이 아닌, sam feature만을 사용해서 qformer에 cross attention시킴 
    white_image_crossattention: bool=True #모델 forward시 qformer가 흰색 이미지일경우 cross attention받을지 말지. 기본은 받는 것

    # Additional loss config
    category_classification_loss: bool=False

    def to_dict(self) -> Dict[str, Any]:
        """
            HF PretrainedConfig's to_dict() make class attribute "model_type" to ignore argparse's model_type.
            (cause trouble when saving model and loading with config)
        """
        output = copy.deepcopy(self.__dict__)
        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = output.pop("_pre_quantization_dtype", None)
        self.dict_torch_dtype_to_str(output)

        return output

@dataclass
class DataArguments:
    split_type: str="PS"
    split_path: str="/data/split"
    data_path: str="/data/SMART101-release-v1/SMART101-Data"
    puzzle_path: str="/data/SMART101-release-v1/puzzle_type_info.csv"

    #for SAM feature
    sam_feature_path: str=field(default=None) #NOTE : 1. 기본은 None sh file에서 지정해줘야함 (dataset 에서 그래야 사용/비사용 구분이 가능) 2. decoder feature 사용시에 path 바꿔줘야함

    # num_class: int=91
    prediction_type: str=field(default="answerkey") #could be ["answerkey","answervalue"]. answerkey predict one of 'A','B','C','D','E'. answervalue predict float/str value
    background_exclude: bool=False

    # Caption 실험 argument
    use_caption: bool=False #caption 실험 
    caption_path: str="/data/QWEN_caption/Qwen_caption.json"

    # SAM token mask 실험
    SAM_token_mask: bool=False  
    token_mask_path: str="/data/SAM_features/decoder_features/token_mask_features"
    
    # category classification loss
    # category_classification_mapping_path = "/data/category_mapping/puzzle_2_categorynum_mapping.json"
    category_classification_mapping_path: str=field(default=None)


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
    max_length=256
