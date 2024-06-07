"""
    return initialized model and processor.
    If model choice is not using processor, return None for processor
"""
from typing import Dict, Optional, Sequence, List
from transformers import PretrainedConfig, GenerationConfig

from models.basemodel import BaseLLMModel
from models.flant5.tokenization_t5 import T5Tokenizer
from models.flant5.configuration_t5 import T5Config

from models.smart_basemodel import SMART_Net


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
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    if "flant5" in model_args.llm_model_type:
        if training_args.llm_load_ckpt_path == None:
            tokenizer = T5Tokenizer.from_pretrained(model_args.llm_pretrained_model_path)
            model = BaseLLMModel(model_args).to(training_args.device)
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_args.llm_pretrained_model_path)
            model_config = T5Config.from_pretrained(training_args.llm_load_ckpt_path)
            model = BaseLLMModel.from_pretrained(pretrained_model_name_or_path=training_args.llm_load_ckpt_path,
                                            config=model_config
                                            ).to(training_args.device)
    else:
        raise NotImplementedError

    return model, tokenizer