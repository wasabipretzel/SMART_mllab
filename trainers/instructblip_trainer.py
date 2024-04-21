"""_summary_
    trainer for training instructblip base model. Specifically, purpose of this trainer is for applying 
    different learning rates for pretrained/scratch modules in model
"""
from transformers import Seq2SeqTrainer
import torch 

class InstructblipTrainer(Seq2SeqTrainer):

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
        scratch_modules = ["lora"]

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