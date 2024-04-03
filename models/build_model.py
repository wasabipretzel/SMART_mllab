"""
    return initialized model and processor.
    If model choice is not using processor, return None for processor
"""
from typing import Dict, Optional, Sequence, List
from models.basemodel import BaseModel
from models.instructblip.processing_instructblip import InstructBlipProcessor


#TODO : model from pretrained 상황인 경우? resume_from_checkpoints..
def get_model(model_args, device):
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
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = BaseModel(model_args).to(device)
    else:
        raise NotImplementedError

    return model, processor