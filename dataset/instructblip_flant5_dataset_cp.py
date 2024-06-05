import os
import math
import pickle as pkl
from tqdm import tqdm

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers

from utils.util import is_float

from itertools import permutations
import random

class InstructblipFlant5Dataset(Dataset):
    """
        single image를 읽고 processing해서 올린다 (instructblip processor어디서 동작하는지 찾아볼것)
        question and answer 밀어올리는 방식은 
    """
    def __init__(self, data_args, mode, processor=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        import pudb; pudb.set_trace()
        self.data_args = data_args
        self.mode = mode
        self.info_file = pd.read_csv(data_args.puzzle_path)
        self.qa_info = self.get_qainfo()
        if data_args.puzzle_type is not None:
            self.qa_info = self.select_qainfo()
        self.generate_option_key()
        self.append_prediction_type() #use for option approximation when prediction type is 'answervalue'
        if data_args.permutation and mode == 'train':
            self.qa_info = self.permutation_option()
        
        """
            single instance example of self.qa_info

            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """

    def permutation_option(self):
        print(self.qa_info[0])
        print('before permutation :', len(self.qa_info))
        permuted_qa_info = []
        for qa_pair in tqdm(self.qa_info):
            options = qa_pair['options'][:-1].split(', ')
            permuted_option_list = [p for p in permutations(options)]
            permuted_option_list = random.sample(permuted_option_list, 3) # 3개 sample
            for permuted_option in permuted_option_list:
                qa_pair['options'] = ', '.join(permuted_option)+'.'
                permuted_qa_info.append(qa_pair)
        print('after permutation :', len(permuted_qa_info))
        print(permuted_qa_info[0])
        return permuted_qa_info

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
        data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, self.mode+'.pkl')
        with open(data_path, 'rb') as f:
            qa_info = pkl.load(f)
        return qa_info

    def select_qainfo(self) -> List[dict]:
        selected_qainfo = []
        for qa in tqdm(self.qa_info):
            pid = int(qa['puzzle_id'])
            if self.info_file[self.info_file['puzzle_id'] == pid].type.item() == self.data_args.puzzle_type:
                selected_qainfo.append(qa)
        return selected_qainfo

    def load_image(self, qa_pair):
        """
            qa_pair -> 
            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
        image_path = os.path.join(self.data_args.data_path, qa_pair["puzzle_id"], "img", qa_pair["image"])
        image = Image.open(image_path).resize((self.data_args.data_image_size, self.data_args.data_image_size)).convert("RGB")

        return image

    def get_input_text(self, qa_pair, pid):
        #process input text -> this function can be developed for instruction tuning etc
        if self.data_args.prediction_type == 'answerkey':
            prompt = "Please read the following question, select the correct answer from one of the options.\n"
        elif self.data_args.prediction_type == 'answervalue':
            prompt = "Please read the following question, calculate the answer value based on the provided options. You should answer only the value.\n"
        if self.data_args.add_cot:
            prompt = "Let’s think step by step. " + prompt
        if self.data_args.add_puzzle_option:
            # print('pid', pid)
            cur_puzzle_type = self.info_file[self.info_file['puzzle_id'] == pid].type.item()
            # print('cur_puzzle_type', cur_puzzle_type)
            prompt = f"This puzzle is about {cur_puzzle_type}. " + prompt

        # else:
        #     raise NotImplementedError

        question = "Question : " + qa_pair["Question"] + '\n' + 'Options : ' + qa_pair["options"] + "\n" + "Answer : "

        return prompt + question

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

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        pid = int(single_qa_pair["puzzle_id"])
        image = self.load_image(single_qa_pair)
        text_input = self.get_input_text(single_qa_pair, pid)
        text_output = self.get_output_text(single_qa_pair) 
        option_values = self.get_option_values(single_qa_pair)
        data = {
            'pixel_values' : image,
            # llm input
            'text_input' : text_input, #prompt + "Question :" + question + "Answer : "
            'text_output': text_output,

            # for qformer instruction input
            'question' : single_qa_pair["Question"],

            # for different collator action
            'mode' : self.mode, 
            # for evaluation
            'pid' : pid,
            'option_values' : option_values,
            'answer_type' : single_qa_pair["answer_type"]
        }

        return data

@dataclass
class InstructblipFlant5_collator(object):
    """Collate examples for supervised fine-tuning."""
    """
        flant5의 경우, input : text, output : text + eos token만 붙히면 나머지는 t5안에서 처리를 해준다. 
        flant5은 right padding이 기본
    """
    processor:transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_pixel_values = []
        b_text_input = []
        b_text_output = []
        b_qformer_text_input = []
        mode = instances[0]["mode"]

        for each_batch in instances:
            #qformer input
            b_pixel_values.append(each_batch["pixel_values"]) 
            b_qformer_text_input.append(each_batch["question"])
            #llm I/O
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])

        #qformer input
        image_input = self.processor(images=b_pixel_values, return_tensors='pt')
        qformer_text_input = self.processor(text=b_qformer_text_input, padding=True, truncation=True, return_tensors='pt')
        #llm I/O 
        text_input = self.processor(text=b_text_input, padding=True, truncation=True, return_tensors='pt')
        #flant5은 항상 right padding이기에 train/test에 따라 padding side신경쓸 필요없음
        #대신 output에 eos token을 끝에 붙혀야하기에 이 부분만 조절
        self.processor.tokenizer.add_eos_token=True
        text_output = self.processor(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
        self.processor.tokenizer.add_eos_token=False
        #target
        targets = text_output.input_ids.masked_fill(
            text_output.input_ids == self.processor.tokenizer.pad_token_id, -100 
        )
        if mode == "train":
            inputs = {
                "pixel_values" : image_input.pixel_values,
                "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
                "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
                "input_ids" : text_input.input_ids,
                "attention_mask" : text_input.attention_mask,
                "decoder_input_ids" : None,
                "decoder_attention_mask" : None,
                "labels" : targets,
            }
        else:
            inputs = {
                "pixel_values" : image_input.pixel_values,
                "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
                "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
                # for generation, need different input_ids and att_mask
                "input_ids" : text_input.input_ids, #t5 encoder input
                "attention_mask" : text_input.attention_mask,
                "labels" : targets,
            } 
        
        return inputs



