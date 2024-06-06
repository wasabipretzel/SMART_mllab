import os
import pickle as pkl
import json

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers

from utils.util import is_float

class Flant5Dataset(Dataset):
    def __init__(self, data_args, mode, tokenizer=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data_args = data_args
        self.mode = mode
        self.qa_info = self.get_qainfo()
        self.generate_option_key()
        self.append_prediction_type() #use for option approximation when prediction type is 'answervalue'

        """
            single instance example of self.qa_info

            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
    
    def generate_option_key(self):
        """_summary_
            given self.qa_info, create option key and value for input text prompt
            {'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9'} -> {options : "A : 6, B : 13, C : 12, D : 10, E : 9"}
        """
        option_candidates = ["A","B","C","D","E"]
        for qa_pair in self.qa_info:
            option_values = ""
            for each_option in option_candidates:
                if each_option != "E":
                    update_msg = f"{each_option} : {qa_pair[each_option]}, "
                else:
                    update_msg = f"{each_option} : {qa_pair[each_option]}."
                option_values += update_msg
            qa_pair["options"] = option_values
        return
    
    def append_prediction_type(self):
        """_summary_
            given self.qa_info, add value whether problem answer type is float/string. (For option approximation)
            method : if all option value can be converted to float, answer type is float. Else string type
            Later, string type will be approximate with embedding cosine similarity. Float type will be approximate with distance measure.
        """
        option_candidates = ["A","B","C","D","E"]
        for qa_pair in self.qa_info:
            float_flag = True
            for each_option in option_candidates:
                if is_float(qa_pair[each_option]) == False:
                    float_flag = False
            if float_flag == True:
                qa_pair["answer_type"] = 'float'
            else:
                qa_pair["answer_type"] = 'string'
        return

    def get_qainfo(self) -> List[dict]:
        """
            load all QA pair & image metadata
        """
        data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, 'train.pkl')
        with open(data_path, 'rb') as f:
            qa_info = pkl.load(f)
        # white 이미지만 거르기 (9, 30, 38, 47, 89, 91) # 일단은 그냥 가져오자
        white_image_index = ["9", "30", "38", "47", "89", "91"]
        test_white_image_index = [self.data_args.test_puzzle_num]
        train_white_image_index = list(set(white_image_index)-set(test_white_image_index))

        # train 5개, test 1개로 분리
        if self.mode == "train":
            qa_info = [qa_info[i] for i in range(len(qa_info)) if qa_info[i]["puzzle_id"] in train_white_image_index]
        else: # test
            qa_info = [qa_info[i] for i in range(len(qa_info)) if qa_info[i]["puzzle_id"] in test_white_image_index]
            
        return qa_info

    def get_input_text(self, qa_pair):
        #process input text -> this function can be developed for instruction tuning etc
        if self.data_args.prediction_type == 'answerkey':
            prompt = "Please read the following question, select the correct answer from one of the options.\n"
        elif self.data_args.prediction_type == 'answervalue':
            prompt = "Please read the following question, calculate the answer value based on the provided options. You should answer only the value.\n"
        else:
            raise NotImplementedError

        question = "Question : " + qa_pair["Question"] + '\n' + 'Options : ' + qa_pair["options"] + "\n" + "Answer : "


        input_text = prompt + question

        return input_text

    def get_output_text(self, qa_pair):
        # Answers can be modified with multi-hop reasoning process
        # answer_prefix = "Answer : "
        if self.data_args.prediction_type == 'answerkey':
            # one of 'A','B','C','D','E'
            answer = qa_pair["Answer"]
        elif self.data_args.prediction_type == 'answervalue':
            # alphabet답을 먼저 구하고 => qa_info에서 그 답을 key로 하면 value가 나옴
            # 만약 answer_type이 float면 답안도 float. type이 string이면 답안도 string
            answer_key = qa_pair["Answer"]
            # if qa_pair["answer_type"] == 'float':
            #     answer = float(qa_pair[answer_key])
            # else:
            #     answer = qa_pair[answer_key]
            answer = qa_pair[answer_key] #float 의미가 없는게 tokenize될때 string이어야 함
        else:
            raise NotImplementedError

        return answer

    def get_option_values(self, qa_pair):
        """_summary_
            given single qa pair, get option values and change it to float/string by answer type.
        Args:
            qa_pair (_type_): _description_
        """
        option_values = []
        opts_candidates=["A","B","C","D","E"]
        for option_key in opts_candidates:
            if qa_pair["answer_type"] == "float":
                option_values.append(float(qa_pair[option_key]))
            else:
                option_values.append(qa_pair[option_key])
        return option_values
    
    # def check_white(self, qa_pair):
    #     image_path = os.path.join(self.data_args.data_path, qa_pair["puzzle_id"], "img", qa_pair["image"])
    #     low, high = Image.open(image_path).convert("L").getextrema() #range of image value
    #     if low == high == 255:
    #         return True
    #     else:
    #         return False

    def get_category_num(self, qa_pair):
        puzzle_id = qa_pair["puzzle_id"]
        category_num = self.puzzle_2_category[puzzle_id]

        return category_num #NOTE need to check dtype

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        text_input = self.get_input_text(single_qa_pair)
        text_output = self.get_output_text(single_qa_pair) 
        pid = int(single_qa_pair["puzzle_id"])
        option_values = self.get_option_values(single_qa_pair)
        
        data = {
            # llm input
            'text_input' : text_input, #prompt + "Question :" + question + "Answer : "
            'text_output': text_output,
            # for different collator action
            'mode' : self.mode, 
            # for evaluation
            'pid' : pid,
            'option_values' : option_values,
            'answer_type' : single_qa_pair["answer_type"],
        }


        return data

@dataclass
class Flant5_collator(object):
    """Collate examples for supervised fine-tuning."""
    """
        flant5의 경우, input : text, output : text + eos token만 붙히면 나머지는 t5안에서 처리를 해준다. 
        flant5은 right padding이 기본
    """
    data_args:transformers.PretrainedConfig
    tokenizer:transformers.T5Tokenizer


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_text_input = []
        b_text_output = []
        mode = instances[0]["mode"]

        for idx, each_batch in enumerate(instances):
            #llm I/O
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])
        #llm I/O 
        text_input = self.tokenizer(text=b_text_input, padding=True, truncation=True, return_tensors='pt')
        #flant5은 항상 right padding이기에 train/test에 따라 padding side신경쓸 필요없음
        #대신 output에 eos token을 끝에 붙혀야하기에 이 부분만 조절
        self.tokenizer.add_eos_token=True
        text_output = self.tokenizer(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
        self.tokenizer.add_eos_token=False
        #target
        targets = text_output.input_ids.masked_fill(
            text_output.input_ids == self.tokenizer.pad_token_id, -100 
        )

        if mode == "train":
            inputs = {
                "input_ids" : text_input.input_ids,
                "attention_mask" : text_input.attention_mask,
                "decoder_input_ids" : None,
                "decoder_attention_mask" : None,
                "labels" : targets,
            }
        else:
            inputs = {
                # for generation, need different input_ids and att_mask
                "input_ids" : text_input.input_ids, #t5 encoder input
                "attention_mask" : text_input.attention_mask,
                "labels" : targets,
            } 
        
        return inputs