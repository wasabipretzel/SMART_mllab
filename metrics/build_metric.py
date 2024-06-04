from metrics.base_criterion import ComputeMetricAnswerValue, ComputeMetricAnswerKey, ComputeMetricAnswerAll
from metrics.smart_basecode_criterion import Criterion


def get_metric(model_args, data_args, tokenizer, embeddings, eval_dataset):
    if "flant5" in model_args.llm_model_type:
        if data_args.prediction_type == "answerkey":
            return ComputeMetricAnswerKey(tokenizer=tokenizer, vicuna_embedding=embeddings, eval_dataset=eval_dataset, puzzle_path=data_args.puzzle_path)
        elif data_args.prediction_type == "answervalue":
            return ComputeMetricAnswerValue(tokenizer=tokenizer, vicuna_embedding=embeddings, eval_dataset=eval_dataset, puzzle_path=data_args.puzzle_path)
        elif data_args.prediction_type == "answerall":
            return ComputeMetricAnswerAll(tokenizer=tokenizer, vicuna_embedding=embeddings, eval_dataset=eval_dataset, puzzle_path=data_args.puzzle_path)
    else:
        raise NotImplementedError