from metrics.base_criterion import ComputeMetricAnswerValue, ComputeMetricAnswerKey
from metrics.smart_basecode_criterion import Criterion


def get_metric(model_args, data_args, processor, embeddings, eval_dataset):
    if "instructblip" in model_args.model_type:
        if data_args.prediction_type == "answerkey":
            return ComputeMetricAnswerKey(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_dataset=eval_dataset, puzzle_path=data_args.puzzle_path)
        elif data_args.prediction_type == "answervalue":
            return ComputeMetricAnswerValue(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_dataset=eval_dataset, puzzle_path=data_args.puzzle_path)
    elif model_args.model_type=="R50_BERT":
        return Criterion()
    else:
        raise NotImplementedError