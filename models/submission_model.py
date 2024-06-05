import sys

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig, GenerationConfig
from transformers.utils import logging
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model
from models.instructblip.configuration_instructblip import InstructBlipConfig
from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration
from models.instructblip.processing_instructblip import InstructBlipProcessor
from models.basemodel import BaseModel
from utils.util import NoWarningFilter
import copy
from torch.nn import CosineSimilarity

logger = logging.get_logger('transformers')
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter()) #To avoid warning msg when generating (max_new_token), add custom filter


class SubmissionModel(PreTrainedModel):
    def __init__(self, config, training_args):
        super().__init__(config)
        self.config = config
        self.training_args= training_args


        #어차피 inference 만 할거잖아
        self.keymodel = self.load_key_model(self.config)
        logger.info("loaded key model")

        self.valuemodel = self.load_value_model(self.config)

        logger.info("loaded value model")


        #for classification
        #processor도 init해야함 
        self.option_value = ['A','B','C','D','E','F','G','H']
        self.keyval_cls_processor = InstructBlipProcessor.from_pretrained(config.pretrained_model_path) #NOTE data_args에도 pretrained_model_path추가해야함 -> 이거 정확히 넣어야함!! offline에서도 동작되도록
        #embedding 필요? #CPU
        self.keyval_embedding = copy.deepcopy(self.keymodel.VLM.language_model.get_input_embeddings()).weight.clone().detach()
        self.cossim = CosineSimilarity(dim=1)
        self.key_type_list = ['logic', 'spatial', 'path', 'counting', 'pattern']
        self.value_type_list = ['math', 'measure', 'algebra']
        self.puzzle_type_dict = {
            "A" : "algebra",
            "B" : "math", 
            "C" : "counting", 
            "D" : "logic",
            "E" : "measure",
            "F" : "path",
            "G" : "pattern",
            "H" : "spatial"
        }


    def get_input_text_for_classification(self, question):
        prompt="There are 8 question types. Based on the given image and question, select the question type from the options.\n"
        
        options_cls = ""
        for each_option in ["A","B","C","D","E","F","G","H"]:
            update_msg = f"{each_option} : {self.puzzle_type_dict[each_option]}, "
            options_cls += update_msg
        options_cls = options_cls[:-2]+'.' #NOTE 이거 뭐임?
        question = "Question : " + question[0] + '\n' + 'Options : ' + options_cls + "\n" + "Answer : "

        return prompt + question


    def get_approximated_puzzle_type(self, puzzle_type, option_value):
        """
            puzzle_type : str
        """
        option_tokenized = self.keyval_cls_processor.tokenizer(text = option_value, padding=True, truncation=False, return_tensors="pt").input_ids.long()
        option_embedded = self.keyval_embedding[option_tokenized].mean(axis=1)

        pred_tokenized = self.keyval_cls_processor.tokenizer(text=[puzzle_type], padding=True, truncation=False, return_tensors="pt").input_ids.long()
        each_pred_embedded = self.keyval_embedding[pred_tokenized].mean(axis=1)

        approximated_option_index = self.cossim(option_embedded, each_pred_embedded).argmax(dim=0) #NOTE 이 부분 확인
        result = option_value[approximated_option_index]

        return result

    def load_key_model(self, model_args):

        # model_name = self.config.pretrained_model_path.split("/")[-1]
        # vlm_config=InstructBlipConfig.from_pretrained(f"/SMART_mllab/models/instructblip/{model_name}.json")
        # vlm = InstructBlipForConditionalGeneration(vlm_config)
        # #peft적용
        # vlm.to(torch.bfloat16)
        # lora_config = LoraConfig(
        #             r = config.lora_r,
        #             lora_alpha=config.lora_alpha,
        #             target_modules=["q", "v"],
        #             lora_dropout=config.lora_dropout,
        #             bias="none",
        #             task_type="SEQ_2_SEQ_LM"
        #         )
        # vlm.language_model = get_peft_model(vlm.language_model, lora_config)
        # vlm = vlm.from_pretrained(self.config.key_model_pretrained_ckpt_path)

        model_config = PretrainedConfig.from_pretrained(self.training_args.load_key_ckpt_path)
        model_config.train_mode = False
        # model = BaseModel.from_pretrained(pretrained_model_name_or_path=training_args.load_ckpt_path,
        model = BaseModel.from_pretrained(pretrained_model_name_or_path=self.training_args.load_key_ckpt_path,
                                    config=model_config
                                    )


        return model

    def load_value_model(self, model_args):
        model_config = PretrainedConfig.from_pretrained(self.training_args.load_value_ckpt_path)
        model_config.train_mode = False
        # model = BaseModel.from_pretrained(pretrained_model_name_or_path=training_args.load_ckpt_path,
        model = BaseModel.from_pretrained(pretrained_model_name_or_path=self.training_args.load_value_ckpt_path,
                                        config=model_config
                                        )

        return model

    def forward(self, return_loss=True, **sample):
        # classify which model to forward 
        cls_text_input = self.get_input_text_for_classification(sample["question"])
        cls_text_input = self.keyval_cls_processor(text=[cls_text_input], padding=True, truncation=True, return_tensors="pt")
        
        #NOTE devicecheck
        qformer_input = self.keyval_cls_processor(text=sample["question"], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            generated_keyval = self.keymodel.VLM.generate(
                pixel_values=sample["pixel_values"].to(self.device),
                qformer_input_ids=qformer_input["qformer_input_ids"].to(self.device),
                qformer_attention_mask= qformer_input["qformer_attention_mask"].to(self.device),
                input_ids= cls_text_input.input_ids.to(self.device),
                attention_mask=cls_text_input.attention_mask.to(self.device),
            )
        #NOTE 여기 generation kwargs어떤식으로 들어가는지 봐야함.. 
        puzzle_type = self.keyval_cls_processor.tokenizer.batch_decode(generated_keyval, skip_special_tokens=True)[0]

        if puzzle_type not in self.option_value:
            puzzle_type = self.get_approximated_puzzle_type(puzzle_type, self.option_value) #one of "A" ~ "H"
        
        keyval_category = ""
        if self.puzzle_type_dict[puzzle_type] in self.key_type_list:
            keyval_category = "key"
        else:
            keyval_category = "val"
        

        #main model forward
        if keyval_category == "key":
            #NOTE 여기서 key_input_text 를 input_ids로 명시해주던가 따로 풀어서 넣어주던가 해야함
            result = self.keymodel(
                pixel_values=sample["pixel_values"],
                qformer_input_ids=sample["qformer_input_ids"],
                qformer_attention_mask=sample["qformer_attention_mask"],
                input_ids=sample["key_input_ids"], #여기가 다름
                attention_mask=sample["key_attention_mask"],
                labels=sample["labels"],
                sam_feature=sample["sam_feature"],
                keyval_pred=keyval_category
            ) #['loss', 'logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']

            #language_model_outputs.logits.shape -> [B, S, vocab]
            # logits은 language_model_outputs.logits과 동일함 
        else:
            result = self.valuemodel(
                pixel_values=sample["pixel_values"],
                qformer_input_ids=sample["qformer_input_ids"],
                qformer_attention_mask=sample["qformer_attention_mask"],
                input_ids=sample["value_input_ids"],
                attention_mask=sample["value_attention_mask"],
                labels=sample["labels"],
                sam_feature=sample["sam_feature"],
                keyval_pred=keyval_category
            )

        
        return result

    @torch.no_grad()
    def generate(self, **sample):
        # classify which model to forward 
        cls_text_input = self.get_input_text_for_classification(sample["question"])
        cls_text_input = self.keyval_cls_processor(text=[cls_text_input], padding=True, truncation=True, return_tensors="pt")
        
        #NOTE devicecheck
        qformer_input = self.keyval_cls_processor(text=sample["question"], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            generated_keyval = self.keymodel.VLM.generate(
                pixel_values=sample["pixel_values"].to(self.device),
                qformer_input_ids=qformer_input["qformer_input_ids"].to(self.device),
                qformer_attention_mask= qformer_input["qformer_attention_mask"].to(self.device),
                input_ids= cls_text_input.input_ids.to(self.device),
                attention_mask=cls_text_input.attention_mask.to(self.device),
            )
        #NOTE 여기 generation kwargs어떤식으로 들어가는지 봐야함.. 
        puzzle_type = self.keyval_cls_processor.tokenizer.batch_decode(generated_keyval, skip_special_tokens=True)[0]

        if puzzle_type not in self.option_value:
            puzzle_type = self.get_approximated_puzzle_type(puzzle_type, self.option_value) #one of "A" ~ "H"
        
        keyval_category = ""
        if self.puzzle_type_dict[puzzle_type] in self.key_type_list:
            keyval_category = "key"
        else:
            keyval_category = "val"


        # 여기가 중요!! 
        # 만약 sample에서 올라온 class pred값이 key -> key model로 넣어서 return 하고
        # value -> value model로 넣어서 return
        #NOTE : batchsize항상 1이어야함!!!! 
        if keyval_category == "key":
            result = self.keymodel.generate(
                pixel_values=sample["pixel_values"],
                qformer_input_ids=sample["qformer_input_ids"],
                qformer_attention_mask=sample["qformer_attention_mask"],
                input_ids=sample["key_input_ids"], #여기가 다름
                attention_mask=sample["key_attention_mask"],
                labels=sample["labels"],
                sam_feature=sample["sam_feature"],
                keyval_pred=keyval_category
            )
            #NOTE modeling_instructblip 안에서 output key값 하나 더 줘야함! (generate 때 key값도 바꿔줘야함) -> 이건 forward에서 바꿔줘야함
        else:
            result = self.valuemodel.generate(
                pixel_values=sample["pixel_values"],
                qformer_input_ids=sample["qformer_input_ids"],
                qformer_attention_mask=sample["qformer_attention_mask"],
                input_ids=sample["key_input_ids"], #여기가 다름
                attention_mask=sample["key_attention_mask"],
                labels=sample["labels"],
                sam_feature=sample["sam_feature"],
                keyval_pred=keyval_category
            )
        
        return result



