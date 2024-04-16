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

from utils.util import concat_text_input_output, generation_concat_text_input_output

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
        """
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
        prompt = "Please read the following question, select the correct answer from one of the options.\n"
        question = "Question : " + qa_pair["Question"] + '\n' + 'Options : ' + qa_pair["options"] + "\n" + "Answer : "

        return prompt + question

    def get_output_text(self, qa_pair):
        # Answers can be modified with multi-hop reasoning process
        # answer_prefix = "Answer : "
        answer = qa_pair["Answer"]

        return answer

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        image = self.load_image(single_qa_pair)
        text_input = self.get_input_text(single_qa_pair)
        text_output = self.get_output_text(single_qa_pair) 
        data = {
            'pixel_values' : image,
            # llm input
            'text_input' : text_input, #prompt + "Question :" + question + "Answer : "
            'text_output': text_output,

            # for qformer instruction input
            'question' : single_qa_pair["Question"],

            # for different collator action
            'mode' : self.mode, 
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
        #NOTE 매 output 문장 끝에 eos token 이 붙어있는지 확인할것 
        text_input = self.processor(text=b_text_input, padding=True, truncation=True, return_tensors='pt')
        self.processor.tokenizer.add_eos_token=True
        self.processor.tokenizer.add_bos_token=False
        text_output = self.processor(text=b_text_output, padding=True, truncation=True, return_tensors='pt')
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
            llm_inputs, answer_labels = generation_concat_text_input_output(
                text_input.input_ids,
                text_input.attention_mask,
                text_output.input_ids,
                text_output.attention_mask,
            )

            inputs = {
                "pixel_values" : image_input.pixel_values,
                "qformer_input_ids" : qformer_text_input["qformer_input_ids"],
                "qformer_attention_mask" : qformer_text_input["qformer_attention_mask"],
                # for generation, need different input_ids and att_mask
                "input_ids" : llm_inputs["input_ids"],
                "attention_mask" : llm_inputs["attention_mask"],
                "labels" : answer_labels,
            } 
        
        return inputs

