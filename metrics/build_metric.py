from metrics.base_criterion import ComputeMetric
from metrics.smart_basecode_criterion import Criterion


def get_metric(model_args, processor):
    if model_args.model_type == "instructblip":
        return ComputeMetric(tokenizer=processor.tokenizer)
    elif model_args.model_type=="R50_BERT":
        return Criterion()
    else:
        raise NotImplementedError