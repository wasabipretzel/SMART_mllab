from dataclasses import dataclass
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel, TrainerCallback
from datasets import load_metric
import torch
from torch.nn import CosineSimilarity
import numpy as np

from utils.util import is_float, read_dataset_info

@dataclass
class ComputeMetricVisual:

    tokenizer: transformers.PreTrainedTokenizer
    vicuna_embedding: transformers.PreTrainedModel
    eval_infos: dict
    puzzle_path: str

    def __post_init__(self):

        # self.vicuna_embedding=self.vicuna_embedding.weight.clone().detach()

        self.b_options = self.eval_infos["option_values"]
        self.b_answer_type = self.eval_infos["answer_type"]
        self.b_pids = self.eval_infos["pid"]

        """
        EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids)
            get all logits, labels after all eval_step
        pred.predictions (얘가 맞는듯) (300, 182)
        pred.label_ids  (얜 죄다 -100) (300, 124)
            predictions (`np.ndarray`): Predictions of the model.
            label_ids (`np.ndarray`): Targets to be matched.
        tokenizer을 넣어줘야하는듯. 
        """
        self.lower_candidates = {
            "a" : "A",
            "b" : "B",
            "c" : "C",
            "d" : "D",
            "e" : "E"
        }
        # self.candidates = ["A","B","C","D","E"]
        self.cossim = CosineSimilarity(dim=1)
        self.puzzles = read_dataset_info(self.puzzle_path) #TODO remove hard coding

    # cf) puzzle metric, 등등을 위해 csv 읽어서 올려야함 
    def compute_metrics(self, pred):
        correct_category_preds = pred.category_predictions.sum().item() #[True, False ...]
        category_accuracy = correct_category_preds / pred.category_predictions.shape[0]

        metrics= {}
        #category acc
        metrics[f"category_acc"] = category_accuracy

        return metrics

    
    def make_submission_json(self):
        return
