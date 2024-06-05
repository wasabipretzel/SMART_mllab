from typing import Dict, Optional, Sequence, List

from dataset.instructblip_vicuna_dataset import InstructblipVicunaDataset, InstructblipVicuna_collator
from dataset.instructblip_flant5_dataset import InstructblipFlant5Dataset, InstructblipFlant5_collator
from dataset.idefics2_dataset import idefics2_dataset, idefics2_collator
from dataset.submission_dataset import SubmissionDataset, SubmissionDataset_collator
from dataset.submission_dynamic_ensemble_dataset import SubmissionEnsembleDataset, SubmissionEnsembleDataset_collator
from dataset.smart_starter import SMART_starter, SMART_starter_collator



def get_dataset(model_args, data_args, mode, processor=None)-> Dict:
    if "instructblip" in model_args.model_type:
        if mode != "test":
            if 'vicuna' in model_args.model_type:
                train_dataset = InstructblipVicunaDataset(data_args=data_args, mode='train')
                val_dataset =  InstructblipVicunaDataset(data_args=data_args, mode='test') # use test set for validation
                data_collator = InstructblipVicuna_collator(processor=processor) 
            elif 'flant5' in model_args.model_type:
                train_dataset = InstructblipFlant5Dataset(data_args=data_args, mode='train')
                val_dataset =  InstructblipFlant5Dataset(data_args=data_args, mode='test') # use test set for validation
                data_collator = InstructblipFlant5_collator(processor=processor) 
            else:
                raise NotImplementedError
        else:
            train_dataset=None
            if data_args.challenge_phase != None:
                if data_args.prediction_type == "ensemble_classify_category":
                    val_dataset = SubmissionEnsembleDataset(data_args=data_args, mode="test")
                    data_collator = SubmissionEnsembleDataset_collator(data_args=data_args, processor=processor)
                else:
                    val_dataset = SubmissionDataset(data_args=data_args, mode="test")
                    data_collator = SubmissionDataset_collator(data_args=data_args, processor=processor)
            elif 'vicuna' in model_args.model_type:
                val_dataset =  InstructblipVicunaDataset(data_args=data_args, mode='test')
                data_collator = InstructblipVicuna_collator(processor=processor) 
            elif 'flant5' in model_args.model_type:
                val_dataset =  InstructblipFlant5Dataset(data_args=data_args, mode='test')
                data_collator = InstructblipFlant5_collator(processor=processor) 
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
