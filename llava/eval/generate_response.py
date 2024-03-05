"""
    GPT eval을 위한 candidate response 생성 파일

    - model initiate
    - checkpoint load
    - eval dataset 생성
    - trainer eval
    - output 받아서 
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
import sys
sys.path.append("/SeqMMLearning")
from llava.model.sequence_mm import SequentialMM_Model
from llava.train.train import *
from peft import PeftModel
from tqdm import tqdm


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_pretrained_ckpt(model_path, model_base, device):
    """
        model_path : 새로 저장된 ckpt 폴더
        model_base : 맨 처음 llava ckpt폴더 (토크나이저 load위해 필요)

        순서 : 기본 llm load -> lora아닌 weight들 load -> lora ckpt 에서 쓸데없는 key값들 제거 -> lora load
    """
    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # 실제 inference시에는 qformer tokenizer도 load해야됨!
    print('Loading LLaVA from base model...')
    llm = LlamaForCausalLM.from_pretrained( # 기본 llm load-> 여기에 lora아닌거, lora인거 붙혀서 load할거임
            model_base,
            config=lora_cfg_pretrained,
            )

    model_args = DotDict({ #학습때 사용했던 arg들 만들어놓음 
        "use_pretrained_qformer" : True,
        "pretrained_qformer_path" : "/data/pretrained_models/qformer_pretrained",
        "pretrained_qformer_tokenizer_path" : "/data/pretrained_models/qformer_pretrained/qformer_tokenizer",
        "pretrained_qformer_query_token_path" : "/data/pretrained_models/qformer_pretrained/query_tokens/query_tokens.pth"

    })
    qf_tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_qformer_tokenizer_path)

    model = SequentialMM_Model(llm=llm, query_num=32, args=model_args, device=device) #깡통 모델 load
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu') 

    model.load_state_dict(non_lora_trainables, strict=False) # lora아닌애들 올리기

    adapter = torch.load('/data/ckpt/adapter_model.bin', map_location='cpu') # lora ckpt load

    adapter = {(k.replace("llm.base_model.model", "model.llm") if 'llm.base_model.' in k else k): v for k, v in adapter.items()} #key 값 이상한거 갈아엎기

    print(f"before peft : {count_parameters(model.llm)}")

    model.llm = PeftModel.from_pretrained(model.llm, model_path) # peft load해서 모델 llm에 붙이기 

    print(f"after peft : {count_parameters(model.llm)}")

    return model, tokenizer, qf_tokenizer

#argument init

def load_eval_dataset(tokenizer, qf_tokenizer):
    data_args = DotDict({ #학습때 사용했던 arg들 만들어놓음 
        "data_path" : "/data/dataset/split/data.json",
        "val_txt_path" : "/data/dataset/split/val.txt",
        "feature_path" : "/data/dataset/features"
    })
    #load하는거  똑같이 한다음에 dataset geneartion
    val_dataset =  LazySupervisedDataset_Assembly(tokenizer=tokenizer,
                                    qf_tokenizer= qf_tokenizer,
                                    data_path=data_args.data_path,
                                    txt_path=data_args.val_txt_path,
                                    data_args=data_args, is_train=False)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, qf_tokenizer=qf_tokenizer)

    return val_dataset, data_collator


def main():
    device='cuda:2'
    model_path = "/data/ckpt"
    model_base = "/SeqMMLearning/checkpoints/llava-v1.5-7b"
    model, tokenizer, qf_tokenizer = load_pretrained_ckpt(model_path, model_base, device)
    model.to(device)
    val_dataset, data_collator = load_eval_dataset(tokenizer, qf_tokenizer)

    val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=1,
                                num_workers=4,
                                shuffle=False,
                                collate_fn = data_collator
                                )
    
    pred =[]
    gt_answer= []
    for idx, sample in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        answer_text = sample["answer_text"][0]
        gt_answer.append(answer_text)
        generate_ids = model.predict(**sample)

        predict_text = tokenizer.batch_decode(generate_ids)[0]
        pred.append(predict_text)
    
    #save
    result = {}
    cnt =0
    for each_pred, each_gt in zip(pred, gt_answer):
        result[cnt] = {
            "gt" : each_gt,
            "pred" : each_pred
        }
        cnt += 1
    
    with open('/data/generate/ft_eval_gen/IKEA_eval_gen.json', 'w') as f:
        json.dump(result, f)

    return 


if __name__ == "__main__":
    main()
