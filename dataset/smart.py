import os
import math
import pickle as pkl

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers

from utils.util import concat_text_input_output, generation_concat_text_input_output, is_float

class SMART(Dataset):
    """
        single image를 읽고 processing해서 올린다 (instructblip processor어디서 동작하는지 찾아볼것)
        question and answer 밀어올리는 방식은 
    """
    def __init__(self, data_args, mode, processor=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data_args = data_args
        self.mode = mode
        # if processor != None:
        #     self.processor = processor
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
        data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, self.mode+'.pkl')
        with open(data_path, 'rb') as f:
            qa_info = pkl.load(f)
        return qa_info

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
        image = Image.open(image_path).convert("RGB")

        return image

    def get_input_text(self, qa_pair):
        #process input text -> this function can be developed for instruction tuning etc
        if self.data_args.prediction_type == 'onlyanswer':
            prompt = "Please read the following question, select the correct answer from one of the options.\n"
        elif self.data_args.prediction_type == 'answervalue':
            prompt = "Please read the following question, calculate the answer value based on the provided options. You should answer only the value.\n"
        else:
            raise NotImplementedError

        question = "Question : " + qa_pair["Question"] + '\n' + 'Options : ' + qa_pair["options"] + "\n" + "Answer : "

        return prompt + question

    def get_output_text(self, qa_pair):
        # Answers can be modified with multi-hop reasoning process
        # answer_prefix = "Answer : "
        if self.data_args.prediction_type == 'onlyanswer':
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
        image = self.load_image(single_qa_pair)
        text_input = self.get_input_text(single_qa_pair)
        text_output = self.get_output_text(single_qa_pair) 
        pid = int(single_qa_pair["puzzle_id"])
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
class SMART_collator(object):
    """Collate examples for supervised fine-tuning."""
    processor:transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_pixel_values = []
        b_text_input = []
        b_text_output = []
        b_qformer_text_input = []
        mode = instances[0]["mode"]

        #for eval
        # b_option_values = []
        # b_answer_type=[]
        # b_pids=[]

        for each_batch in instances:
            #qformer input
            b_pixel_values.append(each_batch["pixel_values"]) 
            b_qformer_text_input.append(each_batch["question"])
            #llm I/O
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])
            # #for eval
            # b_option_values.append(each_batch["option_values"])
            # b_answer_type.append(each_batch["answer_type"])
            # b_pids.append(each_batch["pid"])


        #qformer input
        image_input = self.processor(images=b_pixel_values, return_tensors='pt')
        qformer_text_input = self.processor(text=b_qformer_text_input, padding=True, truncation=True, return_tensors='pt')
        #llm I/O 
        #NOTE 매 output 문장 끝에 eos token 이 붙어있는지 확인할것 
        text_input = self.processor(text=b_text_input, padding=True, truncation=True, return_tensors='pt')
        self.processor.tokenizer.add_eos_token=True
        self.processor.tokenizer.add_bos_token=False
        if mode == "train":
            text_output = self.processor(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
        else:
            self.processor.tokenizer.padding_side="right"
            text_output = self.processor(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
            self.processor.tokenizer.padding_side="left"
        self.processor.tokenizer.add_eos_token=False

        if mode == "train":
            llm_inputs, input_part_targets_len = concat_text_input_output(
                text_input.input_ids,
                text_input.attention_mask,
                text_output.input_ids,
                text_output.attention_mask,
            )
            #target
            targets = llm_inputs["input_ids"].masked_fill(
                llm_inputs["input_ids"] == self.processor.tokenizer.pad_token_id, -100
            )
            for batch_idx, input_length in enumerate(input_part_targets_len):
                targets[batch_idx][:input_length] = -100

            inputs = {
                "pixel_values" : image_input.pixel_values,
                "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
                "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
                #NOTE -> text_input processor거쳐서 무슨 key로 해야 input_ids가 나옴?
                "input_ids" : llm_inputs["input_ids"],
                "attention_mask" : llm_inputs["attention_mask"],
                "labels" : targets,
            }
        else:
            # llm_inputs, answer_labels = generation_concat_text_input_output(
            #     text_input.input_ids,
            #     text_input.attention_mask,
            #     text_output.input_ids,
            #     text_output.attention_mask,
            # )

            # inputs = {
            #     "pixel_values" : image_input.pixel_values,
            #     "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
            #     "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
            #     # for generation, need different input_ids and att_mask
            #     "input_ids" : llm_inputs["input_ids"],
            #     "attention_mask" : llm_inputs["attention_mask"],
            #     "labels" : answer_labels,
            #     #for eval
            #     "eval_ingredients" : [b_option_values, b_answer_type, b_pids] 
            # } 
            inputs = {
                "pixel_values" : image_input.pixel_values,
                "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
                "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
                # for generation, need different input_ids and att_mask
                "input_ids" : text_input.input_ids,
                "attention_mask" : text_input.attention_mask,
                "labels" : text_output.input_ids,
                #for eval
                # "eval_ingredients" : [b_option_values, b_answer_type, b_pids] 
            } 
        
        return inputs



