from dataclasses import dataclass
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel, TrainerCallback
from datasets import load_metric
import torch
from torch.nn import CosineSimilarity
import numpy as np

from utils.util import is_float, read_dataset_info

@dataclass
class SubmissionCriterionAnswerKey:

    tokenizer: transformers.PreTrainedTokenizer
    vicuna_embedding: transformers.PreTrainedModel
    eval_infos: dict
    puzzle_path: str

    def __post_init__(self):

        self.vicuna_embedding=self.vicuna_embedding.weight.clone().detach()
        
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
    # method
    # 1. pred 같이 밀어올리는 부분은 self.qa_info의 answer type, options(option들의 값), pids 
    # 2. pred.predictions을 pad_token_id제외하고 batch_decode
    # 3. answer_list만들고 for문돌면서 float가능한지 아니면 str로 분류해야하는지 판단
    # 4. (1) 만약 pred값이랑 answer value랑 같아 -> s_acc을 위해 cache.
    #    (2) 다른 경우 : 만약 problem도 float고 pred도 float면 distance기반 option approximate
    #    (3) problem이랑 상관없이 pred가 string이면 embedding approximate
    #    (4) argmax해서 가장 유사한 답안으로 pred을 다시 만들기 
    # 5. pred값으로 s_acc구하고 o_acc도 구하기 + puzzle별로 + class별로도 구하기 

    # cf) puzzle metric, 등등을 위해 csv 읽어서 올려야함 
    def compute_metrics(self, pred):
        #b_options : nested list of [test_samples, 5]
        #b_answer_type : List[str] (len:num test samples)
        #b_pids : List[str] (len: num test samples)
        #method 1
        #method 2
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id #fill -100 index with pad_token_id (preventing index/overflow error)
        gt_answer_list = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True) #get rid of pad tokens

        pred.predictions[pred.predictions == -100] = self.tokenizer.pad_token_id
        pred_answer_list = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        #위의 값들이 detach되어있는지 확인할 것
        # tokenizing 시 맨 앞 bos token안붙히게 할 것
        self.tokenizer.add_bos_token=False
        non_approximated_pred = []
        approximated_pred = [] # List[bool] : len=num test samples (근사하고 맞았는지 틀렸는지)
        approximated_answer = [] #List[str] : 실제 답으로 선정할 option key 값
        print(f"pred answer list : {len(pred_answer_list)}")
        for idx, each_pred in enumerate(pred_answer_list):
            each_pred = str(each_pred).strip() #생성 답에 '4\n'이런게 있음 
            gt_answer = gt_answer_list[idx]
            #problem answer type에 상관없이 답이 ABCD가 아니다 -> embedding기반으로 비교
            #소문자인경우 대문자로 변경
            if each_pred in self.lower_candidates.keys():
                each_pred = self.lower_candidates[each_pred]
            #for s_acc
            if each_pred == gt_answer:
                non_approximated_pred.append(True)
            else:
                non_approximated_pred.append(False)
            
            #approximation
            #'A','B','C','D','E' 와 비교
            option_value = ['A','B','C','D','E']
            option_tokenized = self.tokenizer(text = option_value, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[5, seqlen, 4096]
            #NOTE : padding시 padding token의 embedding은 어떻게 되는지 보기
            option_embedded = self.vicuna_embedding[option_tokenized].mean(axis=1) #[5, 4096]

            each_pred_tokenized = self.tokenizer(text=each_pred, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[1, seqlen, 4096]
            each_pred_embedded = self.vicuna_embedding[each_pred_tokenized].mean(axis=1) #[1, 4096]

            approximated_option_index = self.cossim(option_embedded, each_pred_embedded).argmax(dim=0)
            result = (option_value[approximated_option_index] == gt_answer)
            approximated_pred.append(result)
            approximated_answer.append(option_value[approximated_option_index])

        metrics = {
            "predicted_answers" : approximated_answer,
            "pids" : self.b_pids
        }
        #원상복구
        self.tokenizer.add_bos_token=True

        return metrics

    
    def make_submission_json(self):
        return

@dataclass
class SubmissionCriterionAnswerValue:
    
    tokenizer: transformers.PreTrainedTokenizer
    vicuna_embedding: transformers.PreTrainedModel
    eval_infos: dict
    puzzle_path: str

    def __post_init__(self):

        self.vicuna_embedding=self.vicuna_embedding.weight.clone().detach()

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
        self.candidates = ["A","B","C","D","E"]
        self.cossim = CosineSimilarity(dim=1)
    # method
    # 1. pred 같이 밀어올리는 부분은 self.qa_info의 answer type, options(option들의 값), pids 
    # 2. pred.predictions을 pad_token_id제외하고 batch_decode
    # 3. answer_list만들고 for문돌면서 float가능한지 아니면 str로 분류해야하는지 판단
    # 4. (1) 만약 pred값이랑 answer value랑 같아 -> s_acc을 위해 cache.
    #    (2) 다른 경우 : 만약 problem도 float고 pred도 float면 distance기반 option approximate
    #    (3) problem이랑 상관없이 pred가 string이면 embedding approximate
    #    (4) argmax해서 가장 유사한 답안으로 pred을 다시 만들기 
    # 5. pred값으로 s_acc구하고 o_acc도 구하기 + puzzle별로 + class별로도 구하기 

    # cf) puzzle metric, 등등을 위해 csv 읽어서 올려야함 
    def compute_metrics(self, pred):
        #b_options : nested list of [test_samples, 5]
        #b_answer_type : List[str] (len:num test samples)
        #b_pids : List[str] (len: num test samples)
        #method 1
        #method 2
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id #fill -100 index with pad_token_id (preventing index/overflow error)
        gt_answer_list = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True) #get rid of pad tokens

        pred.predictions[pred.predictions == -100] = self.tokenizer.pad_token_id
        pred_answer_list = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        
        #위의 값들이 detach되어있는지 확인할 것
        # tokenizing 시 맨 앞 bos token안붙히게 할 것
        self.tokenizer.add_bos_token=False
        non_approximated_pred = []
        approximated_pred = [] # List[bool] : len=num test samples (근사하고 맞았는지 틀렸는지)
        approximated_answer = [] #List[str] : 실제 답으로 선정할 option key 값
        answer_key = ["A","B","C","D","E"]
        for idx, each_pred in enumerate(pred_answer_list):
            each_pred = each_pred.strip() #생성 답에 '4\n'이런게 있음 
            #method 3
            problem_answer_type = self.b_answer_type[idx] #'string' or 'float'
            #gt preprocessing
            if problem_answer_type == 'float':
                gt_answer = float(gt_answer_list[idx])
            else:
                # 혹시 모를 앞뒤 공백같은 것 없애기 NOTE : 이런 조취에도 corner case생길 경우를 위해 많이 찍어보기!!!
                gt_answer = gt_answer_list[idx].strip()
            #pred val preprocessing
            if is_float(each_pred):
                each_pred = float(each_pred)
            else:
                each_pred = each_pred.strip()

            #method 4.1
            if gt_answer == each_pred:
                non_approximated_pred.append(True)
            else:
                non_approximated_pred.append(False)
            #method 4.2
            if problem_answer_type=='float' and is_float(each_pred):
                option_value = self.b_options[idx] #[1, 5]
                approximated_index = np.abs(np.array(option_value).astype("float") - torch.tensor(each_pred).unsqueeze(0).numpy()).argmin(axis=0)
                result = (option_value[approximated_index] == gt_answer)
                # result = (np.abs(np.array(option_value).astype("float") - torch.tensor(each_pred).unsqueeze(1).numpy()).argmin(axis=1)
                #             == gt_answer) #NOTE axis=1이 맞나? result type보고 안에 빼야하는지 확인, argmin하면 index가 나오지않나? -> gt_answer가 index을 담은 것...인데 아니면 확인
                approximated_pred.append(result)
                approximated_answer.append(answer_key[approximated_index])
            else:
                #둘 중 하나라도 string인 경우
                #method 4.3
                option_value = self.b_options[idx]
                option_value = [str(opt_val) for opt_val in option_value] #pred가 str이고 option 이 float인경우 안하면 tokenizer에서 문제가 생김
                option_tokenized = self.tokenizer(text = option_value, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[5, seqlen, 4096]
                #NOTE : padding시 padding token의 embedding은 어떻게 되는지 보기
                option_embedded = self.vicuna_embedding[option_tokenized].mean(axis=1) #[5, 4096]

                each_pred = str(each_pred)
                each_pred_tokenized = self.tokenizer(text=each_pred, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[1, seqlen, 4096]
                each_pred_embedded = self.vicuna_embedding[each_pred_tokenized].mean(axis=1) #[1, 4096]

                approximated_option_index = self.cossim(option_embedded, each_pred_embedded).argmax(dim=0)
                result = (option_value[approximated_option_index] == gt_answer)
                approximated_pred.append(result)
                approximated_answer.append(answerkey[approximated_option_index])

        metrics = {
            "predicted_answers" : approximated_answer,
            "pids" : self.b_pids
        }
        
        #원상복구
        self.tokenizer.add_bos_token=True

        return metrics

    
    def make_submission_json(self):
        return


@dataclass
class SubmissionCriterionAnswerAll:
    
    tokenizer: transformers.PreTrainedTokenizer
    vicuna_embedding: transformers.PreTrainedModel
    eval_infos: dict
    puzzle_path: str

    def __post_init__(self):

        self.vicuna_embedding=self.vicuna_embedding.weight.clone().detach()

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
        self.candidates = ["A","B","C","D","E"]
        self.cossim = CosineSimilarity(dim=1)
    # method
    # 1. pred 같이 밀어올리는 부분은 self.qa_info의 answer type, options(option들의 값), pids 
    # 2. pred.predictions을 pad_token_id제외하고 batch_decode
    # 3. answer_list만들고 for문돌면서 float가능한지 아니면 str로 분류해야하는지 판단
    # 4. (1) 만약 pred값이랑 answer value랑 같아 -> s_acc을 위해 cache.
    #    (2) 다른 경우 : 만약 problem도 float고 pred도 float면 distance기반 option approximate
    #    (3) problem이랑 상관없이 pred가 string이면 embedding approximate
    #    (4) argmax해서 가장 유사한 답안으로 pred을 다시 만들기 
    # 5. pred값으로 s_acc구하고 o_acc도 구하기 + puzzle별로 + class별로도 구하기 

    # cf) puzzle metric, 등등을 위해 csv 읽어서 올려야함 
    def compute_metrics(self, pred):
        #b_options : nested list of [test_samples, 5]
        #b_answer_type : List[str] (len:num test samples)
        #b_pids : List[str] (len: num test samples)
        #method 1
        #method 2
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id #fill -100 index with pad_token_id (preventing index/overflow error)
        gt_answer_list = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True) #get rid of pad tokens

        pred.predictions[pred.predictions == -100] = self.tokenizer.pad_token_id
        pred_answer_list = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        
        #위의 값들이 detach되어있는지 확인할 것
        # tokenizing 시 맨 앞 bos token안붙히게 할 것
        self.tokenizer.add_bos_token=False
        approximated_answer = []
        # 1. option value들을 "A : 3" 이런식으로 재구성해주기
        for idx, each_pred in enumerate(pred_answer_list):
            each_pred = each_pred.strip() #생성 답에 '4\n'이런게 있음 
            #method 3
            problem_answer_type = self.b_answer_type[idx] #'string' or 'float'


            #gt preprocessing
            gt_answer = gt_answer_list[idx].strip()
            gt_key = gt_answer.split(" : ")[0]  #-> "A" or "B" ..

            #option preprocessing
            each_options = []
            for each_key, each_option_val in zip(self.candidates, self.b_options[idx]):
                each_options.append(f"{each_key} : {each_option_val}")
            

            option_tokenized = self.tokenizer(text = option_value, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[5, seqlen, 4096]
            option_embedded = self.vicuna_embedding[option_tokenized].mean(axis=1) #[5, 4096]

            each_pred = str(each_pred)
            each_pred_tokenized = self.tokenizer(text=each_pred, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[1, seqlen, 4096]
            each_pred_embedded = self.vicuna_embedding[each_pred_tokenized].mean(axis=1) #[1, 4096]

            approximated_option_index = self.cossim(option_embedded, each_pred_embedded).argmax(dim=0)
            result_option_key = each_options[approximated_option_index].split(" : ")[0]

            approximated_answer.append(result_option_key)
        

        metrics = {
            "predicted_answers" : approximated_answer,
            "pids" : self.b_pids
        }
        
        #원상복구
        self.tokenizer.add_bos_token=True

        return metrics

    
    def make_submission_json(self):
        return