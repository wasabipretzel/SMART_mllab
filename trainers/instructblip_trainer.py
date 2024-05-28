"""_summary_
    trainer for training instructblip base model. Specifically, purpose of this trainer is for applying 
    different learning rates for pretrained/scratch modules in model
"""
from typing import Dict
from transformers import Seq2SeqTrainer
import torch 

class InstructblipTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.category_loss = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        #============custom multiple loss =======================
        cls_loss = outputs["category_loss"]
        if cls_loss is not None:
            loss += cls_loss
        
        # Log the separate losses
        # self.log({'main_loss': outputs.loss.item()})
        if cls_loss is not None:
            self.category_loss = cls_loss.item()
            # self.log({'cls_loss': cls_loss.item()})

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        In instructblip base model, 
        pretrained module parts are ["qformer", "language_projection"] (need to use very small lr)
        scratch module parts are ["llm lora", "qformer lora"(not implemented), "vit adapter"(not implemented)] (need to use relatively large lr)

        #TODO : 나중에 qformer에 lora붙히고 난 이후에 lora제외한 qformer와 나머지 qformer을 분리할 수 있도록 디자인해야함
                => vision adapter도 마찬가지
        #TODO : pretrained_parameters, scratch_parameters에서 bias은 어떻게 처리해주는게 맞지?
        """ 
        opt_model = self.model
        pretrained_modules = ["qformer", "language_projection"]
        scratch_modules = ["lora", "sam_linear", "category_cls_head"]

        if self.args.pretrained_module_lr is not None and self.args.scratch_module_lr is not None: 
            pretrained_parameters, scratch_parameters= list(), list()
            for name, param in opt_model.named_parameters():
                for module in pretrained_modules:
                    if module in name and param.requires_grad:
                        pretrained_parameters.append(param)
                for module in scratch_modules:
                    if module in name and param.requires_grad:
                        scratch_parameters.append(param)

            optimizer_grouped_parameters = [
                {'params': pretrained_parameters, 'lr': self.args.pretrained_module_lr},
                {'params': scratch_parameters, 'lr': self.args.scratch_module_lr}
            ]

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        return self.optimizer


    def log(self, logs:Dict[str, float]):
        """
            logs : {'loss': 4.2812, 'learning_rate': 7.042253521126761e-11}
            -> learning_rate을 빼고 pretrained_lr, scratch_lr로 보고하도록..
        """
        if 'eval' in list(logs.keys())[0]:
            super().log(logs)
        else:
            key_names = ["pretrained_lr", "scratch_lr"]
            step = logs.get("step", 0)
            if "learning_rate" in logs.keys():
                del logs["learning_rate"]
            lr_dict = {
                f'{key_names[i]}': group['lr'] for i, group in enumerate(self.optimizer.param_groups)
            }
            if self.category_loss != None:
                lr_dict["category_loss"] = self.category_loss
            logs.update(lr_dict)
            super().log(logs)

