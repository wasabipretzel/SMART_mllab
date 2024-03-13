import json
import os

import torch 
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
from llava.config.hf_config import *
from llava.utils import rank0_print
from llava import conversation as conversation_lib
"""
    source code for dataset and datacollator class for HF trainer
"""

def preprocess_assembly(
    sources: Sequence[str],
    has_image: bool = False
) -> Dict:
    """
        prompt + Question : question + Answer : answer 
        가 있을 때 text_input, text_output을 return 

        text_input = 'prompt + Question : question + Answer :'
        text_output = 'answer'
    """
    #prompt : <im_st> <im_end> sys ### question ### answer
    header = f"{conversation_lib.default_conversation.system}\n\n" # general prompt 
    # conversation = _add_speaker_and_signal_assembly(header, sources) # 1. Add signal
    conversation = header + sources
    # conversations = [conversation]
    question_sep = "Question"
    answer_sep = "Answer"

    #text_input
    text_input = conversation.split(answer_sep)[0]

    text_output = answer_sep + conversation.split(answer_sep)[-1]

    # if text_output[0] != ':':
    #     text_input + ' : '
    # else: # ':' 가 이미 붙어 있는 경우 input 마지막에 'Answer ' 가 되도록 하여 full text가  'Answer : step 9 ~' 이렇게 되도록 한다
    #     text_input + ' '
    
    result = {
        "text_input" : text_input, #str
        "text_output" : text_output
    }

    return result


class LazySupervisedDataset_Assembly(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str, 
                 txt_path: str,
                 data_args: DataArguments,
                 is_train: bool):
        super(LazySupervisedDataset_Assembly, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # txt_file = open(txt_path, "r")

        self.all_data = list_data_dict
        self.txt_path = txt_path
        self.data_args = data_args
        self.feature_path = self.data_args.feature_path
        self.is_train=is_train

        with open(self.txt_path, 'r') as f:
            candidates = f.readlines() #[11, 32, 3, 0, ...] #학습 또는 val 에 사용할 후보군들
        self.candidates = [idx.strip() for idx in candidates]

    def __len__(self):
        return len(self.candidates)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
            input_ids : List[] -> sequence length
            qformer_input_ids : List[] -> question length
            target 
            "im_st" (input_ids)
            "im_end" (input_ids)
            "feat" [num_img, 576]
            "num_img" : int
        """
        #load text feature
        search_key = self.candidates[idx]
        single_case = self.all_data[search_key] #{id : ~, content : ~}
        serial_number = single_case["id"]
        content = single_case["content"]

        # input_ids, qformer_input_ids, target, im_st, im_end
        data_dict = preprocess_assembly(
            content,    
            has_image=False
        ) #input_ids with chunks


        #load image feature
        single_feature_path = os.path.join(self.feature_path, serial_number+'.npy')
        feat = torch.tensor(np.load(single_feature_path)) #[num_img, 576]
        num_img = feat.shape[0]
        img_getitem = {
            "feat" : feat, #tensor
            "num_img" : num_img, #int
            
        }

        data_dict.update(img_getitem)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #text
        #image
        b_feat = []
        b_feat_mask = []
        b_img_nums = []
        b_text_input = []
        b_text_output = []
        #get largest image num in batch
        for each_batch in instances:
            b_img_nums.append(each_batch["num_img"])
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])
        b_max_img = max(b_img_nums)

        _, num_token, D = instances[0]["feat"].shape 
        #pad text first

        #pad image feats
        for each_batch in instances:
            if each_batch["feat"].shape[0] == b_max_img:
                b_feat.append(each_batch["feat"])
                att_mask = torch.ones((b_max_img, num_token))
                b_feat_mask.append(att_mask)
            else:
                #부족한 만큼 padding 
                each_feat = each_batch["feat"] #[num_img, num_token, D]
                att_mask = torch.ones((each_feat.shape[0], num_token)) #[num_img, num_token]
                marginal = torch.zeros((b_max_img - each_feat.shape[0], num_token, D))
                att_mask_zero = torch.zeros((b_max_img - each_feat.shape[0], num_token))
                each_feat = torch.cat([each_feat, marginal], dim=0)
                att_mask = torch.cat([att_mask, att_mask_zero], dim=0)
                b_feat.append(each_feat)
                b_feat_mask.append(att_mask)

        b_feat = torch.stack(b_feat) #[B, max_img, num_token, D]
        b_feat_mask = torch.stack(b_feat_mask, dim=0) #[B, max_img, num_token]이 되도록

        result = {
            "text_input" : b_text_input,
            "text_output" : b_text_output,

            "image_feat" : b_feat,
            "images_att" : b_feat_mask,
            "image_num_in_batch" : b_img_nums
        
        }

        # images = [instance['images'] for instance in instances]
        # batch['images'] = images 
        return result
