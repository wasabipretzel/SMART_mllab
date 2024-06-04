from metrics.base_criterion import ComputeMetricAnswerValue, ComputeMetricAnswerKey
from metrics.submission_criterion import SubmissionCriterionAnswerKey, SubmissionCriterionAnswerValue
from metrics.cls_criterion import ComputeMetricVisual
from metrics.smart_basecode_criterion import Criterion


def get_metric(model_args, data_args, processor, embeddings, eval_infos):
    if "instructblip" in model_args.model_type:
        if data_args.challenge_phase != None:
            if data_args.prediction_type == "answerkey":
                return SubmissionCriterionAnswerKey(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_infos=eval_infos, puzzle_path=data_args.puzzle_path)
            elif data_args.prediction_type == "answervalue":
                return SubmissionCriterionAnswerValue(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_infos=eval_infos, puzzle_path=data_args.puzzle_path)
            else:
                raise NotImplementedError
        elif data_args.prediction_type == "answerkey":
            return ComputeMetricAnswerKey(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_infos=eval_infos, puzzle_path=data_args.puzzle_path)
        elif data_args.prediction_type == "answervalue":
            return ComputeMetricAnswerValue(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_infos=eval_infos, puzzle_path=data_args.puzzle_path)
    elif model_args.model_type=="R50_BERT":
        return Criterion()
    elif model_args.model_type=="visual_classifier":
        return ComputeMetricVisual(tokenizer=processor.tokenizer, vicuna_embedding=embeddings, eval_infos=eval_infos, puzzle_path=data_args.puzzle_path)
    else:
        raise NotImplementedError