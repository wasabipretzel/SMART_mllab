from dataclasses import dataclass
import transformers
from transformers import PreTrainedTokenizer
from datasets import load_metric

@dataclass
class ComputeMetric:
    tokenizer: transformers.PreTrainedTokenizer
    """
    EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids)
        get all logits, labels after all eval_step
       pred.predictions (얘가 맞는듯) (300, 182)
       pred.label_ids  (얜 죄다 -100) (300, 124)
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
       tokenizer을 넣어줘야하는듯. 
    """
    metric = load_metric("accuracy")
    candidates = {
        "A" : 0,
        "B" : 1,
        "C" : 2,
        "D" : 3,
        "E" : 4,
    }
    def compute_metrics(self, pred):
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id #fill -100 index with pad_token_id (preventing index/overflow error)
        gt_answer_list = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True) #get rid of pad tokens
        #prediction 
        pred.predictions[pred.predictions == -100] = self.tokenizer.pad_token_id
        pred_answer_list = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

        gt_filtered = []
        pred_filtered = []
        for gt, pred_ans in zip(gt_answer_list, pred_answer_list):
            gt_flag=False
            pred_ans_flag=False
            for each_option in self.candidates.keys():
                if each_option in gt and gt_flag == False:
                    gt_filtered.append(self.candidates[each_option])
                    gt_flag=True
                if each_option in pred_ans and pred_ans_flag == False:
                    pred_filtered.append(self.candidates[each_option])
                    pred_ans_flag=True 
            # pred에 아예 A,B,C,D,E 없는 경우
            if pred_ans_flag == False:
                pred_filtered.append(-1)
        
        metrics = self.metric.compute(references=gt_filtered, predictions=pred_filtered)

        return metrics