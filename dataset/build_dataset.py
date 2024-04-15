from typing import Dict, Optional, Sequence, List

from dataset.smart import SMART, SMART_collator
from dataset.smart_starter import SMART_starter, SMART_starter_collator



def get_dataset(model_args, data_args, processor=None)-> Dict:
    if model_args.model_type=="instructblip":
        train_dataset = SMART(data_args=data_args, mode='train')
        val_dataset =  SMART(data_args=data_args, mode='val')
        data_collator = SMART_collator(processor=processor) 

    elif model_args.model_type=="R50_BERT":
        train_dataset = SMART_starter(data_args=data_args,mode='train', processor=processor)
        val_dataset =  SMART_starter(data_args=data_args,mode='val', processor=processor)
        data_collator = SMART_starter_collator() 
    else:
        raise NotImplementedError
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
