import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import math
import pickle as pkl
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers


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
        if processor != None:
            self.processor = processor
        self.qa_info = self.get_qainfo() 
        """
            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
        breakpoint()
        

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


    def onehot_multilabel(self, idxs, classes):
        one_hot = np.zeros(classes)
        one_hot[idxs] = 1
        return one_hot

    def get_target(self, vid):
        # Answers can be modified with multi-hop reasoning process

        return

    def process_text(self, qa_pair):
        #process input text -> this function can be developed for instruction tuning etc
        question = qa_pair["Question"]
        answer = qa_pair["Answer"]
        
        prompt = "Please read the following question, select the correct answer from one of the options.\n"
        question = "Question : " + question

        return

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        image = self.load_image(single_qa_pair)
        inputs = self.process_text(single_qa_pair)
        target = self.get_target(single_qa_pair) #nparray (91,)
        data = {
            'vid' : vid_name,
            'vid_input' : vid_input,
            'target': torch.tensor(target) #[91] tensor
        }

        return data




@dataclass
class SMART_collator(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_vid_names = []
        b_vid_inputs = []
        b_target = []

        for each_batch in instances:
            b_vid_names.append(each_batch["vid"])
            b_vid_inputs.append(each_batch["vid_input"].squeeze(0))
            b_target.append(each_batch["target"])

        #target은 stack
        b_vid_inputs = torch.stack(b_vid_inputs) #[B, T, C, H, W]
        b_target = torch.stack(b_target) #[B, class]

        result = {
            "vid_names" : b_vid_names,
            "vid_input" : b_vid_inputs,
            "labels" : b_target,
        }


        return result

