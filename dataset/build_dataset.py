from typing import Dict, Optional, Sequence, List

from dataset.flant5_dataset import Flant5Dataset, Flant5_collator

from dataset.smart_starter import SMART_starter, SMART_starter_collator



def get_dataset(model_args, data_args, mode, tokenizer=None)-> Dict:
    if "flant5" in model_args.llm_model_type:
        if mode != "test":
            train_dataset = Flant5Dataset(data_args=data_args, mode='train')
            val_dataset = Flant5Dataset(data_args=data_args, mode='test')
            data_collator = Flant5_collator(data_args = data_args, tokenizer=tokenizer) 

        else:
            train_dataset=None
            val_dataset = Flant5Dataset(data_args=data_args, mode='test')
            data_collator = Flant5_collator(data_args = data_args, tokenizer=tokenizer) 
    else:
        raise NotImplementedError
        
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
