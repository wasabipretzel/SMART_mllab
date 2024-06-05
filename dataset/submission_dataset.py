import os
import math
import pickle as pkl
import json

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers

from utils.util import is_float

class SubmissionDataset(Dataset):
    def __init__(self, data_args, mode, processor=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']


        self.data_args = data_args
        self.mode = mode
        self.submission_root = "/dataset/"
        self.image_path = os.path.join(self.submission_root, "test-images")
        self.puzzle_file = "VLAR-val.json" if data_args.challenge_phase == 'val' else 'VLAR-test.json'
        self.qa_info = self.get_qainfo()

        self.qa_info = self.generate_option_key(self.qa_info)
        self.qa_info  = self.append_prediction_type(self.qa_info) #use for option approximation when prediction type is 'answervalue'

        if data_args.add_data and mode == 'train':
            add_data = self.process_add_data()
            add_data = self.generate_option_key(add_data)
            add_data = self.append_prediction_type(add_data)
            self.qa_info = self.qa_info + add_data

        """
            single instance example of self.qa_info

            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
        print('total puzzle num :', len(self.qa_info))

    def process_add_data(self):

        res = []
        res += add_dataset.mathverse_qa_dict 
        res += add_dataset.mathvision_qa_dict
        res += add_dataset.iconqa_qa_dict
        res += add_dataset.scienceqa_qa_dict
        res += add_dataset.mathvista_qa_dict
        res += add_dataset.mmstar_qa_dict
        res += add_dataset.mmbench_qa_dict
        res += add_dataset.raven_qa_dict
        res += add_dataset.mmmu_qa_dict
        return res

        return permuted_qa_info

    def generate_option_key(self, qa_info):
        """_summary_
            given self.qa_info, create option key and value for input text prompt
            {'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9'} -> {options : "A : 6, B : 13, C : 12, D : 10, E : 9"}
        """
        option_candidates = ["A","B","C","D","E","F","G","H"]
        for qa_pair in qa_info:
            option_values = ""
            for each_option in option_candidates: # 추가된 데이터에 대한 작업 (2지선다, 3지선다...)
                # if each_option != "E":
                #     update_msg = f"{each_option} : {qa_pair[each_option]}, "
                # else:
                #     update_msg = f"{each_option} : {qa_pair[each_option]}."
                if each_option in qa_pair:
                    update_msg = f"{each_option} : {qa_pair[each_option]}, "
                    option_values += update_msg
                else:
                    break
            option_values = option_values[:-2]+'.'
            qa_pair["options"] = option_values
        
        return qa_info
    
    def append_prediction_type(self, qa_info):
        """_summary_
            given self.qa_info, add value whether problem answer type is float/string. (For option approximation)
            method : if all option value can be converted to float, answer type is float. Else string type
            Later, string type will be approximate with embedding cosine similarity. Float type will be approximate with distance measure.
        """
        option_candidates = ["A","B","C","D","E","F","G","H"]
        for qa_pair in qa_info:
            float_flag = True
            for each_option in option_candidates:
                if each_option in qa_pair:
                    if is_float(qa_pair[each_option]) == False:
                        float_flag = False
            if float_flag == True:
                qa_pair["answer_type"] = 'float'
            else:
                qa_pair["answer_type"] = 'string'
        return qa_info

    def get_qainfo(self) -> List[dict]:
        """
            load all QA pair & image metadata
        """
        with open(os.path.join(self.submission_root, self.puzzle_file)) as test_file:
            puzzle_data = test_file.read()
        puzzle_data = json.loads(puzzle_data)
        assert('VLAR' in puzzle_data.keys())
        return puzzle_data['VLAR'][1:]

    def select_qainfo(self) -> List[dict]:
        selected_qainfo = []
        for qa in tqdm(self.qa_info):
            pid = int(qa['puzzle_id'])
            if self.info_file[self.info_file['puzzle_id'] == pid].type.item() == self.data_args.puzzle_type:
                selected_qainfo.append(qa)
        return selected_qainfo

    def split_qainfo(self) -> List[dict]:
        if self.data_args.prediction_type == 'answerkey':
            puzzle_type_list = ['logic', 'spatial', 'path']
        elif self.data_args.prediction_type == 'answervalue':
            puzzle_type_list = ['math', 'measure', 'algebra', 'counting', 'pattern']

        selected_qainfo = []
        for qa in tqdm(self.qa_info):
            pid = int(qa['puzzle_id'])
            if self.info_file[self.info_file['puzzle_id'] == pid].type.item() in puzzle_type_list:
                selected_qainfo.append(qa)
        return selected_qainfo

    def split_qainfo_option(self) -> List[dict]:
        selected_qainfo = []
        if self.data_args.prediction_type == 'answerkey':
            for qa in tqdm(self.qa_info):
                if qa['answer_type'] == 'string':
                    selected_qainfo.append(qa)
        elif self.data_args.prediction_type == 'answervalue':
            for qa in tqdm(self.qa_info):
                if qa['answer_type'] == 'float':
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
        if 'puzzle_id' in qa_pair:
            image_path = os.path.join(self.image_path, qa_pair["Image"])
        else:
            image_path = os.path.join(self.image_path, qa_pair["Image"])
        try:
            image = Image.open(image_path).convert("RGB").resize((self.data_args.data_image_size, self.data_args.data_image_size), resample=Image.BICUBIC).convert("RGB") # 좌우 크기 다른 이미지 존재
        except:
            image = qa_pair['Image'] # tmp
        return image

    def get_input_text(self, qa_pair, pid):
        #process input text -> this function can be developed for instruction tuning etc
        if self.data_args.prediction_type == 'answerkey':
            prompt = "Please read the following question, select the correct answer from one of the options.\n"
        elif self.data_args.prediction_type == 'answervalue':
            prompt = "Please read the following question, calculate the answer value based on the provided options. You should answer only the value.\n"
        if self.data_args.add_cot:
            prompt = "Let’s think step by step. " + prompt
        if self.data_args.add_puzzle_option and pid is not None:
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
        opts_candidates=["A","B","C","D","E","F","G","H"]
        for option_key in opts_candidates:
            if option_key in qa_pair:
                # print('qa_pair', qa_pair) # comment by yjheo (24/05/25)
                if qa_pair["answer_type"] == "float":
                    option_values.append(float(qa_pair[option_key]))
                else:
                    option_values.append(qa_pair[option_key])
        return option_values

    def option_permutation(self, qa_pair): # 문자가 규칙적이지 않음 options = 'A : 0,2 and 2, B : 1,2 and 9, C : 2, 4 and 9.' 여기서 C처럼 2, 4
        options = qa_pair['options'][:-1]
        keys = ['A : ', 'B : ', 'C : ', 'D : ', 'E : ', 'F : ', 'G : ', 'H : ']
        option_keys = []
        for key in keys:
            if key in options:
                option_keys.append(key[0])
            options = options.replace(key, '<sep>')
        options = options.replace(', <sep>', '<sep>')
        option_values = options.split('<sep>')[1:]

        if option_keys == option_values: # A : A, B : B, C : C, D : D, E : E 같은 경우
            pass

        else:
            if self.data_args.permutation_option == 'opt-shift':
                options = []
                key_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                random.shuffle(option_values)
                for i, value in enumerate(option_values):
                    if key_list[i] in qa_pair:
                        options.append(f'{key_list[i]} : {value}')
                        qa_pair[key_list[i]] = value
                        if str(value) == str(qa_pair['AnswerValue']):
                            qa_pair['Answer'] = key_list[i]
                    else:
                        break
                qa_pair['options'] = ', '.join(options)+'.'

            elif self.data_args.permutation_option == 'opt-reverse':
                random.shuffle(options)
                qa_pair['options'] = ', '.join(options)+'.'

        return qa_pair
            

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        pid = int(single_qa_pair["Id"])
        if pid == None:
            raise NotImplementedError
        image = self.load_image(single_qa_pair)
        try:
            if self.data_args.permutation:
                single_qa_pair = self.option_permutation(single_qa_pair)
        except:
            print('single_qa_pair', single_qa_pair)
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
class SubmissionDataset_collator(object):
    """Collate examples for supervised fine-tuning."""
    """
        flant5의 경우, input : text, output : text + eos token만 붙히면 나머지는 t5안에서 처리를 해준다. 
        flant5은 right padding이 기본
    """
    data_args:transformers.PretrainedConfig
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

        # qformer input
        try :
            image_input = self.processor(images=b_pixel_values, return_tensors='pt')
        except:
            print('b_pixel_values', b_pixel_values)
            print('b_text_input', b_text_input)
        qformer_text_input = self.processor(text=b_qformer_text_input, padding=True, truncation=True, return_tensors='pt')
        #llm I/O 
        text_input = self.processor(text=b_text_input, padding=True, truncation=True, return_tensors='pt')
        #flant5은 항상 right padding이기에 train/test에 따라 padding side신경쓸 필요없음
        #대신 output에 eos token을 끝에 붙혀야하기에 이 부분만 조절
        self.processor.tokenizer.add_eos_token=True
        text_output = self.processor(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
        self.processor.tokenizer.add_eos_token=False
        # target
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