import os
import shutil
import json
import numpy as np
import torch



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
#     input_part_targets_len = []
#     llm_tokens = {"input_ids": [], "attention_mask": []}
#     for i in range(input_ids.size(0)):
#         breakpoint()
#         this_input_ones = input_atts[i].sum()
#         input_part_targets_len.append(this_input_ones)
#         llm_tokens['input_ids'].append(
#             torch.cat([
#                 input_ids[i][:this_input_ones],
#                 output_ids[i][1:], # to get rid of eos token attached in output_ids (tokenizer automatically attached it)
#                 input_ids[i][this_input_ones:]
#             ])
#         )
#         llm_tokens['attention_mask'].append(
#             torch.cat([
#                 input_atts[i][:this_input_ones],
#                 output_atts[i][1:],
#                 input_atts[i][this_input_ones:]
#             ])
#         )
#     llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
#     llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
#     return llm_tokens, input_part_targets_len


# since training/inference all pad text left, modify this code. (Lavis version pad right in training, pad left in inference)
def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_output_ones = output_ids[i].shape[0] - output_atts[i].sum() # target의 padding길이(단답이 아닐 경우를 대비함)
        # input_part_targets_len.append(this_output_ones + input_atts[i].sum())
        input_part_targets_len.append(this_output_ones+input_ids[i].shape[0]) #총 llm input 중 target전까지
        llm_tokens['input_ids'].append(
            torch.cat([
                output_ids[i][:this_output_ones], #target 앞에 붙은 padding을 input 맨 앞에
                input_ids[i], #already left padded
                output_ids[i][this_output_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                output_atts[i][:this_output_ones],
                input_atts[i],
                output_atts[i][this_output_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len



def generation_concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    """_summary_
    위 함수와 다른 점 : input에 answer가 붙어있지 않지만 answer의 padding을 question앞으로 당기는 것으로 input을 구성 (학습과 동일 setting)
    target은 answer만 있으면 되기 때문에 answer_labels라는 변수 이름으로 명명하여 return 
    Args:
        input_ids (_type_): _description_
        input_atts (_type_): _description_
        output_ids (_type_): _description_
        output_atts (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 
    llm_tokens = {"input_ids": [], "attention_mask": []}
    answer_labels = []
    for i in range(input_ids.size(0)):
        this_output_ones = output_ids[i].shape[0] - output_atts[i].sum() 
        llm_tokens['input_ids'].append(
            torch.cat([
                output_ids[i][:this_output_ones], #target 앞에 붙은 padding을 input 맨 앞에
                input_ids[i], #already left padded
                #generation input에는 answer필요없으므로 제거
            ])
        )
        answer_labels.append(
            torch.cat([ #same with right padding
                output_ids[i][this_output_ones:],
                output_ids[i][:this_output_ones] #padding to be located at right side
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                output_atts[i][:this_output_ones],
                input_atts[i],
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    answer_labels = torch.stack(answer_labels)
    return llm_tokens, answer_labels