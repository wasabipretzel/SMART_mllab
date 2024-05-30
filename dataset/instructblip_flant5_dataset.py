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
from transformers import SamModel, SamProcessor, SamImageProcessor
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

class InstructblipFlant5Dataset(Dataset):
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
        self.generate_option_key()
        self.append_prediction_type() #use for option approximation when prediction type is 'answervalue'

        # for evaluation metric, submission때도 b_pids은 필요해서 있어야 함
        if mode != "train":
            self.eval_infos = {
                "option_values" : [],
                "answer_type" : [],
                "pid" : []
            }
            for single_qa_pair in self.qa_info:
                self.eval_infos["option_values"].append(self.get_option_values(single_qa_pair))
                self.eval_infos["answer_type"].append(single_qa_pair["answer_type"])
                self.eval_infos["pid"].append(int(single_qa_pair["puzzle_id"]))

        if self.data_args.use_caption:
            self.caption_info = self.load_extracted_caption()
        
        if self.data_args.category_classification_mapping_path != None:
            self.puzzle_2_category = self.load_category_mapping()

        if self.data_args.use_dynamic_sam_decoder or self.data_args.use_dynamic_sam_encoder: #TODO : sam-vit-huge ckpt 미리 저장해두기 
            self.device=f"cuda:{self.data_args.local_rank}"
            sam_pretrained_model_path = self.data_args.sam_pretrained_model_path if self.data_args.sam_pretrained_model_path is not None else "facebook/sam-vit-huge"
            self.SAM_image_processor = SamImageProcessor.from_pretrained(sam_pretrained_model_path)
            self.SAM_model = SamModel.from_pretrained(sam_pretrained_model_path).to(self.device)
            self.SAM_model.eval()


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

    def load_extracted_caption(self):
        with open(self.data_args.caption_path, 'r') as f:
            extracted_caption = json.load(f)
        return extracted_caption
    
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
        if self.data_args.prediction_type == 'answerkey':
            prompt = "Please read the following question, select the correct answer from one of the options.\n"
        elif self.data_args.prediction_type == 'answervalue':
            prompt = "Please read the following question, calculate the answer value based on the provided options. You should answer only the value.\n"
        else:
            raise NotImplementedError

        question = "Question : " + qa_pair["Question"] + '\n' + 'Options : ' + qa_pair["options"] + "\n" + "Answer : "

        if self.data_args.use_caption:
            image_name = qa_pair["image"]
            caption_value = self.caption_info[image_name]["caption"]
            # if len(caption_value.split()) <= self.caption_threshold:
            #     caption = "Caption : " + caption_value+'\n'
            # else:
            #     caption_value = " ".join(caption_value.split()[:self.caption_threshold])
            #     caption = "Caption : " + caption_value+'\n'
            caption = "Caption : " + caption_value+'\n'
            input_text= prompt + caption + question
        else:
            input_text = prompt + question # cdddddaptddddddddionddddd

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

    def get_sam_feature(self, qa_pair):
        """
            given single qa pair, load pre-extracted sam feature
            sam encoder feature : [256, 1280] (vit-h)
        """
        image_name = qa_pair["image"].split('.png')[0]
        single_sam_feat_path = os.path.join(self.data_args.sam_feature_path, image_name+'.npy')
        try:
            sam_feat = torch.tensor(np.load(single_sam_feat_path))
        except:
            print(f"{image_name}")
        return sam_feat

    def extract_sam_decoder_feat(self, image): #NOTE : device setting
        """
            given single loaded and converted to RGB channel image, get decoder dense image feature via pretrained SAM-huge model
            image will be upscaled to (1024, 1024), then output to [256, 256] feature
        """
        points_per_batch=64
        return_upscaled_embedding=True
        target_size = self.SAM_image_processor.size["longest_edge"]
        crop_boxes, grid_points, cropped_images, input_labels = self.SAM_image_processor.generate_crop_boxes(
            image, target_size
        )
        model_inputs = self.SAM_image_processor(images=cropped_images, return_tensors="pt")

        with torch.no_grad():
            image_embeddings = self.SAM_model.get_image_embeddings(model_inputs.pop("pixel_values").to(self.device))
            model_inputs["image_embeddings"] = image_embeddings
            model_inputs["return_upscaled_embedding"] = return_upscaled_embedding
        
        n_points = grid_points.shape[1]
        
        each_image_attentions= []
        for i in range(0, n_points, points_per_batch):
            batched_points = grid_points[:, i : i + points_per_batch, :, :].to(self.device)
            labels = input_labels[:, i : i + points_per_batch].to(self.device)
            is_last = i == n_points - points_per_batch 

            each_input = {
                "input_points" : batched_points,
                "input_labels" : labels,
                "input_boxes" : crop_boxes,
                "is_last" : is_last,
                **model_inputs,
            } 

            input_boxes = each_input.pop("input_boxes")
            is_last = each_input.pop("is_last")
            original_sizes = each_input.pop("original_sizes").tolist()
            reshaped_input_sizes = each_input.pop("reshaped_input_sizes").tolist()

            model_outputs = self.SAM_model(**each_input, output_attentions=True)
            # model upscaled_embeddings : [B, patchnum, query, 65536]
            # low_res_masks : [1, 64, 3, 256, 256]
            # mask_decoder_attentions[-1] -> [64, 1, 4096, 256]
            # 1. [64, 1, 64*64, 256] -> [64, 1, 16*16, 256] -> [1, 1, 16*16, 256] -> 16
            mask_decoder_attentions = model_outputs["mask_decoder_attentions"][-1].squeeze(1) #[64, 64*64, 256]
            # [64, 256, 64*64] 가 되도록 permute
            mask_decoder_attentions = mask_decoder_attentions.permute(0, 2, 1) #[64, 256, 64*64]
            mask_decoder_attentions = mask_decoder_attentions.reshape(points_per_batch, 256, 64, 64)
            sequence_pooled_attentions = F.avg_pool2d(mask_decoder_attentions, kernel_size=4, stride=4) #[64, 256, 16, 16]
            sequence_pooled_attentions = sequence_pooled_attentions.reshape(points_per_batch, 256, -1).permute(0, 2, 1) #[64, 16*16, 256]
            pooled_attentions = torch.mean(sequence_pooled_attentions, dim=0) #[16*16, 256]
            each_image_attentions.append(pooled_attentions.detach().cpu())
        each_image_attentions = torch.stack(each_image_attentions) #[16, 16*16, 256] -> 1024을 64로 나눈것이니..
        each_image_attentions = torch.mean(each_image_attentions, dim=0) #[256, 256]

        return each_image_attentions

    def extract_sam_encoder_feat(self, image):
        model_inputs = self.SAM_image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_embeddings = self.SAM_model.get_image_embeddings(model_inputs.pop("pixel_values").to(self.device))
            #NOTE need to check the size 
        return image_embeddings

    
    def check_white(self, qa_pair):
        image_path = os.path.join(self.data_args.data_path, qa_pair["puzzle_id"], "img", qa_pair["image"])
        low, high = Image.open(image_path).convert("L").getextrema() #range of image value
        if low == high == 255:
            return True
        else:
            return False
    
    def get_token_mask(self, qa_pair):
        feature_path = os.path.join(self.data_args.token_mask_path, qa_pair["image"]+'.npy')

        #만약 이 feature path에 존재하지 않으면 SAM mask가 뽑히지 않았거나 완전 흰색이거나...-> 그냥 전부 attend하게끔 함
        if not os.path.isfile(feature_path):
            token_mask = torch.ones([257]).long()
        else:
            token_mask = torch.tensor(np.load(feature_path)).unsqueeze(0).long() #[1, 224, 224]
            token_mask = F.avg_pool2d(token_mask, kernel_size=14, stride=14).squeeze(0) #[16, 16]
            token_mask = (token_mask != 0).long()
            token_mask = token_mask.reshape(-1) #[256]
            token_mask = torch.cat([torch.ones([1]), token_mask])
        
        return token_mask #[257]

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
        text_input = self.get_input_text(single_qa_pair)
        text_output = self.get_output_text(single_qa_pair) 
        pid = int(single_qa_pair["puzzle_id"])
        option_values = self.get_option_values(single_qa_pair)

        # for sam feature
        if self.data_args.sam_feature_path != None:
            sam_feature = self.get_sam_feature(single_qa_pair)
        else:
            sam_feature = None

        if self.data_args.use_dynamic_sam_decoder == True:
            sam_feature = self.extract_sam_decoder_feat(image)
        elif self.data_args.use_dynamic_sam_encoder == True:
            sam_feature = self.extract_sam_encoder_feat(image)

        if self.data_args.SAM_token_mask:
            image_attention_mask = self.get_token_mask(single_qa_pair)
        
        if self.data_args.category_classification_mapping_path != None:
            gt_category_num = self.get_category_num(single_qa_pair)
        
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
            'answer_type' : single_qa_pair["answer_type"],

            # for SAM feature experiment
            'sam_feature' : sam_feature,

            # for white background exclude
            'is_only_white' : self.check_white(single_qa_pair), #bool
            "image_attention_mask" : image_attention_mask if self.data_args.SAM_token_mask else None,
            
            # for additional category loss
            "category_num" : gt_category_num if self.data_args.category_classification_mapping_path != None else None
        }

        return data

@dataclass
class InstructblipFlant5_collator(object):
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
        b_sam_feature = []
        white_image_index = []
        token_mask_image_attention = [] if self.data_args.SAM_token_mask else None
        b_category_gt_num = [] if self.data_args.category_classification_mapping_path != None else None
        mode = instances[0]["mode"]

        for idx, each_batch in enumerate(instances):
            #qformer input
            b_pixel_values.append(each_batch["pixel_values"]) 
            b_qformer_text_input.append(each_batch["question"])
            #llm I/O
            b_text_input.append(each_batch["text_input"])
            b_text_output.append(each_batch["text_output"])
            #sam feature
            b_sam_feature.append(each_batch["sam_feature"])
            if each_batch["is_only_white"]:
                white_image_index.append(idx)
            if self.data_args.SAM_token_mask:
                token_mask_image_attention.append(each_batch["image_attention_mask"])
            if self.data_args.category_classification_mapping_path != None:
                b_category_gt_num.append(each_batch["category_num"])


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

        #sam feature
        b_sam_feature = torch.stack(b_sam_feature, dim=0) if b_sam_feature[0] != None else None
        if self.data_args.SAM_token_mask:
            token_mask_image_attention = torch.stack(token_mask_image_attention, dim=0)

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
                #for sam experiment
                "sam_feature" : b_sam_feature, # if sam feature is used, torch.tensor, else None -> need to used at modeling_instructblip.py
                "white_image_index" : white_image_index,
                "image_attention_mask" : token_mask_image_attention if self.data_args.SAM_token_mask else None,
                "category_gt" : torch.tensor(b_category_gt_num) if b_category_gt_num != None else None 
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
                #for sam experiment
                "sam_feature" : b_sam_feature,
                "white_image_index" : white_image_index,
                "image_attention_mask" : token_mask_image_attention if self.data_args.SAM_token_mask else None,
            } 
        
        return inputs



