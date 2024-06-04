"""
    Category classification model
    1. Multimodal Embedding model
    2. Visual Feature model
    3. Textual feature model
"""
import torch
import copy
import torch.nn as nn 
import transformers 
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig, GenerationConfig
from transformers.utils import logging
from typing import Dict, Optional, Sequence, List
from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration
from models.instructblip.configuration_instructblip import InstructBlipConfig


class CategoryClassfier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 128, 32]):
        super(CategoryClassfier, self).__init__()
        # Create a list of all sizes which includes input, hidden, and output layers
        sizes = [input_size] + hidden_sizes + [output_size]
        # List comprehension to create layers based on sizes
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        self.activation = nn.ReLU()  # You can change the activation function here

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        # Output layer without activation function
        x = self.layers[-1](x)
        return x


class VisualClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.train_mode == True:
            self.VLM = InstructBlipForConditionalGeneration.from_pretrained(config.pretrained_model_path)
            #deep copy vision model
            self.vision_model = copy.deepcopy(self.VLM.vision_model)
            for param in self.vision_model.parameters():
                param.requires_grad=False
            del self.VLM
            self.category_cls_head = CategoryClassfier(input_size=config.visual_classifier.hidden_size, output_size=8)
        else:
            model_name = self.config.pretrained_model_path.split("/")[-1]
            self.vlm_config=InstructBlipConfig.from_pretrained(f"/SMART_mllab/models/instructblip/{model_name}.json")
            self.VLM = InstructBlipForConditionalGeneration(self.vlm_config)

            self.vision_model = self.VLM.vision_model.copy.deepcopy()
            del self.VLM
            self.category_cls_head = CategoryClassfier(input_size=config.visual_classifier.hidden_size, output_size=8)
        

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
    ):
        
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values = pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        cls_embeds=vision_outputs.pooler_output #[B, 1408]

        # cls_embeds = image_embeds[:, 0, :] #should be select cls tokens per batch #[B, 1408] #NOTE check dimension of cls_embeds

        cls_loss=None
        correct_predictions=None
        
        category_hidden_state = self.category_cls_head(cls_embeds) #[B, 8]
        # multiclass classification -> softmax
        probabilities = nn.Softmax(dim=1)(category_hidden_state)
        category_predictions = torch.argmax(probabilities, dim=1)
        correct_predictions = (category_predictions == labels) #GT를 compute_metrics에 넣을 수 없어 여기서 미리 맞는/틀린거 넘긴다.
        cls_loss_criterion = nn.CrossEntropyLoss(reduction="mean")  # The loss function #NOTE test this loss to be focal loss!!
        cls_loss = cls_loss_criterion(category_hidden_state, labels)

        results = {
            "loss" : cls_loss,
            "logits" : category_hidden_state,
            "category_predictions" : correct_predictions
        }
        return results



        

        


# class TextualClassifier(PreTrainedModel):
#     """
#         given question, encode bert model. Then, 
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         if self.config.train_mode == True:
#             self.VLM = InstructBlipForConditionalGeneration.from_pretrained(config.pretrained_model_path)
#             #deep copy vision model
#             self.vision_model = self.VLM.vision_model.copy.deepcopy()
#             del self.VLM
#             self.category_cls_head = CategoryClassfier(input_size=config.visualclassifier.hidden_size, output_size=8)
#         else:
#             model_name = self.config.pretrained_model_path.split("/")[-1]
#             self.vlm_config=InstructBlipConfig.from_pretrained(f"/SMART_mllab/models/instructblip/{model_name}.json")
#             self.VLM = InstructBlipForConditionalGeneration(self.vlm_config)

#             self.vision_model = self.VLM.vision_model.copy.deepcopy()
#             del self.VLM
#             self.category_cls_head = CategoryClassfier(input_size=config.visualclassifier.hidden_size, output_size=8)
        

#     def forward(
#         self,
#         pixel_values: torch.FloatTensor,
#         output_attentions: Optional[bool]=None,
#         output_hidden_states: Optional[bool]=None,
#         return_dict: Optional[bool]=None,
#     )

#         vision_outputs = self.vision_model(
#             pixel_values = pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             retrun_dict=return_dict,
#         )

#         image_embeds=vision_outputs[0] #[B, 257, 1408]

#         cls_embeds = image_embeds[:, 0, :] #should be select cls tokens per batch #[B, 1408] #NOTE check dimension of cls_embeds

#         cls_loss=None
#         correct_predictions=None
        
#         category_hidden_state = self.category_cls_head(cls_embeds) #[B, 8]
#         # multiclass classification -> softmax
#         probabilities = nn.Softmax(dim=1)(category_hidden_state)
#         category_predictions = torch.argmax(probabilities, dim=1)
#         correct_predictions = (category_predictions == category_gt) #GT를 compute_metrics에 넣을 수 없어 여기서 미리 맞는/틀린거 넘긴다.
#         cls_loss_criterion = nn.CrossEntropyLoss(reduction="mean")  # The loss function #NOTE test this loss to be focal loss!!
#         cls_loss = cls_loss_criterion(category_hidden_state, category_gt)

#         results = {
#             "loss" : cls_loss,
#             "category_predictions" : correct_predictions
#         }
