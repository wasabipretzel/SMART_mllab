"""
    return initialized model and processor.
    If model choice is not using processor, return None for processor
"""
from typing import Dict, Optional, Sequence, List
from transformers import PretrainedConfig

from models.basemodel import BaseModel
from models.instructblip.processing_instructblip import InstructBlipProcessor

from models.smart_basemodel import SMART_Net


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
    elif model_args.model_type == "R50_BERT":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.DEFAULT
        vision_model = resnet50(weights=weights)
        processor = weights.transforms()
        if model_args.smart_starter_pretrained_path != None:
            checkpoint = torch.load(args.smart_starter_pretrained_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                    # remove prefix
                    state_dict[k[len("module.encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = vision_model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        model = SMART_Net(vision_model)
    else:
        raise NotImplementedError

    return model, processor