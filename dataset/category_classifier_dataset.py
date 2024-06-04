import os
import math
import pickle as pkl
import json

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers
from torchvision.transforms import Resize  

from utils.util import is_float

"""
    single instance example of self.qa_info

    {'id': '1', 
    'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
    'image': 'puzzle_19_e_1.png', 
    'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
    'Answer': 'D', 'Note': 'C(5|2)', 
    'puzzle_id': '19', 'AnswerValue': 10}
"""

class VisualCategoryClassifierDataset(Dataset):
    """
        single image를 읽고 processing해서 올린다 (instructblip processor어디서 동작하는지 찾아볼것)
        question and answer 밀어올리는 방식은 
    """
    def __init__(self, data_args, mode, processor=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data_args = data_args
        self.mode = mode
        self.qa_info = self.get_qainfo()

        # for evaluation metric, submission때도 b_pids은 필요해서 있어야 함
        if mode != "train":
            self.eval_infos = {
                "option_values" : [],
                "answer_type" : [],
                "pid" : []
            }
        
        if self.data_args.category_classification_mapping_path != None:
            self.puzzle_2_category = self.load_category_mapping()


    def load_category_mapping(self):
        """
            {   "puzzle_id (str) " : category num (0~7)
                "1" : 0,
                "2" : 3,
                ..
            }
        """
        with open(self.data_args.category_classification_mapping_path, 'r') as f:
            puzzle_id_2_category_num = json.load(f)
        return puzzle_id_2_category_num


    def get_qainfo(self) -> List[dict]:
        """
            load all QA pair & image metadata
        """
        if self.mode == "train":
            if self.data_args.use_train_legacy:
                data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, self.mode+'_legacy.pkl')
            else:
                data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, self.mode+'.pkl')
        else:
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



    def get_category_num(self, qa_pair):
        image_name = qa_pair["image"]  # 'puzzle_19_e_1.png', 
        puzzle_id = qa_pair["puzzle_id"]
        category_num = self.puzzle_2_category[puzzle_id]

        return category_num #NOTE need to check dtype

    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        image = self.load_image(single_qa_pair)
        pid = int(single_qa_pair["puzzle_id"])

        
        if self.data_args.category_classification_mapping_path != None:
            gt_category_num = self.get_category_num(single_qa_pair)
        
        data = {
            'pixel_values' : image,

            # for different collator action
            'mode' : self.mode, 
            
            # for additional category loss
            "category_num" : gt_category_num if self.data_args.category_classification_mapping_path != None else None
        }

        return data

@dataclass
class VisualCategoryClassifier_collator(object):
    """Collate examples for supervised fine-tuning."""
    """
        flant5의 경우, input : text, output : text + eos token만 붙히면 나머지는 t5안에서 처리를 해준다. 
        flant5은 right padding이 기본
    """
    data_args:transformers.PretrainedConfig
    processor:transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_pixel_values = []
        b_category_gt_num = [] if self.data_args.category_classification_mapping_path != None else None
        mode = instances[0]["mode"]

        for idx, each_batch in enumerate(instances):
            #qformer input
            b_pixel_values.append(each_batch["pixel_values"]) 
            if self.data_args.category_classification_mapping_path != None:
                b_category_gt_num.append(each_batch["category_num"])
        
        image_input = self.processor(images=b_pixel_values, return_tensors='pt')

        inputs = {
            "pixel_values" : image_input.pixel_values,
            "labels" : torch.tensor(b_category_gt_num),
        }
        
        return inputs



