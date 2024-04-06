"""
    return initialized model and processor.
    If model choice is not using processor, return None for processor
"""
from typing import Dict, Optional, Sequence, List
from transformers import PretrainedConfig

from models.basemodel import BaseModel
from models.instructblip.processing_instructblip import InstructBlipProcessor


#TODO : model from pretrained 상황인 경우? resume_from_checkpoints..
def get_model(model_args, training_args):
    """_summary_
        function for initialize model and processor. Processor will be used in dataset
    Args:
        model_args (_type_): need for which model to initialize
        device (_type_): gpu device

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: initialized model or processor
    """
    if model_args.model_type == "instructblip":
        if training_args.load_ckpt_path == None:
            processor = InstructBlipProcessor.from_pretrained(model_args.pretrained_model_path)
            model = BaseModel(model_args).to(training_args.device)
        else:
            processor = InstructBlipProcessor.from_pretrained(model_args.pretrained_model_path)
            model_config = PretrainedConfig.from_pretrained(training_args.load_ckpt_path)
            model = BaseModel.from_pretrained(pretrained_model_name_or_path=training_args.load_ckpt_path,
                                            config=model_config
                                            ).to(training_args.device)
    else:
        raise NotImplementedError

    return model, processor