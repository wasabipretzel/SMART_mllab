from typing import Dict, Optional, Sequence, List

from dataset.instructblip_vicuna_dataset import InstructblipVicunaDataset, InstructblipVicuna_collator
from dataset.instructblip_flant5_dataset import InstructblipFlant5Dataset, InstructblipFlant5_collator
from dataset.instructblip_flant5_dataset_dynamic import InstructblipFlant5DynamicDataset, InstructblipFlant5Dynamic_collator
from dataset.submission_dataset import SubmissionDataset, SubmissionDataset_collator
from dataset.smart_starter import SMART_starter, SMART_starter_collator



def get_dataset(model_args, data_args, mode, processor=None)-> Dict:
    if "instructblip" in model_args.model_type:
        if mode != "test":
            if 'vicuna' in model_args.model_type:
                train_dataset = InstructblipVicunaDataset(data_args=data_args, mode='train')
                val_dataset =  InstructblipVicunaDataset(data_args=data_args, mode='val')
                data_collator = InstructblipVicuna_collator(processor=processor) 
            elif 'flant5' in model_args.model_type:
                if data_args.use_dynamic_caption or data_args.use_dynamic_sam_decoder or data_args.use_dynamic_sam_encoder:
                    train_dataset = InstructblipFlant5DynamicDataset(data_args=data_args, mode='train')
                    val_dataset =  InstructblipFlant5DynamicDataset(data_args=data_args, mode='test')
                    data_collator = InstructblipFlant5Dynamic_collator(data_args = data_args, processor=processor) 
                else:
                    train_dataset = InstructblipFlant5Dataset(data_args=data_args, mode='train')
                    val_dataset =  InstructblipFlant5Dataset(data_args=data_args, mode='test')
                    data_collator = InstructblipFlant5_collator(data_args = data_args, processor=processor) 
            else:
                raise NotImplementedError
        else:
            train_dataset=None
            if data_args.challenge_phase != None:
                val_dataset = SubmissionDataset(data_args=data_args, mode="test")
                data_collator = SubmissionDataset_collator(data_args=data_args, processor=processor)
            elif 'vicuna' in model_args.model_type:
                val_dataset =  InstructblipVicunaDataset(data_args=data_args, mode='test')
                data_collator = InstructblipVicuna_collator(processor=processor) 
            elif 'flant5' in model_args.model_type:
                if data_args.use_dynamic_caption or data_args.use_dynamic_sam_decoder or data_args.use_dynamic_sam_encoder:
                    val_dataset =  InstructblipFlant5DynamicDataset(data_args=data_args, mode='test')
                    data_collator = InstructblipFlant5Dynamic_collator(data_args = data_args, processor=processor) 
                else:
                    val_dataset =  InstructblipFlant5Dataset(data_args=data_args, mode='test')
                    data_collator = InstructblipFlant5_collator(data_args = data_args, processor=processor) 
            else:
                raise NotImplementedError
    elif model_args.model_type=="R50_BERT":
        train_dataset = SMART_starter(data_args=data_args,mode='train', processor=processor)
        val_dataset =  SMART_starter(data_args=data_args,mode='test', processor=processor)
        data_collator = SMART_starter_collator() 
    else:
        raise NotImplementedError


    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
