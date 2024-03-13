"""
    source file for huggingface Argparser.
    Arguments classes must inherit dataclasses.
"""
from dataclasses import dataclass, field
import transformers
from transformers import TrainingArguments, PretrainedConfig
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments(PretrainedConfig):
    model_name_or_path: Optional[str] = field(default="/SeqMMLearning/checkpoints/llava-v1.5-7b")
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    #lora config
    lora_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    version: Optional[str] = field(default="sequential_reasoning")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_model_path: Optional[str] = field(default='/SeqMMLearning/checkpoints/llava-v1.5-7b/mm_projector.bin')

    #patch mamba
    mamba_patch_d_model: int=768
    mamba_patch_d_state: int=4
    mamba_patch_d_conv: int=4
    mamba_patch_expand: int=2

    #mamba_img_seq
    mamba_img_seq_d_model: int=768
    mamba_img_seq_d_state: int=4
    mamba_img_seq_d_conv: int=2
    mamba_img_seq_expand: int=2


@dataclass
class DataArguments:
    data_path: str = field(default="/data/dataset/split/data.json",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'pad'
    train_txt_path: str = '/data/dataset/split/train.txt'
    val_txt_path: str = '/data/dataset/split/val.txt'
    feature_path: str = '/data/dataset/features'

    


@dataclass
class TrainingArguments(TrainingArguments):
    """
        trainingarguments을 상속받았기 때문에 num_train_epochs, per_device_train_batch_size등이 자동으로 들어감 
    """
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    mm_projector_lr: Optional[float] = None

    group_by_modality_length: bool = field(default=False)
    freeze_pretrained: bool=field(default=False)
