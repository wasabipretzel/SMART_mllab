import os
import shutil
import json
import numpy as np
import torch
import logging


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                #NOTE : 이걸 해주면 answer중에 padding이 다른 경우에는 문제가 됨. generation때는 그냥 input은 그대로 넣으면 되고 answer도 
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
    try:
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    except:
        breakpoint()
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    answer_labels = torch.stack(answer_labels)
    return llm_tokens, answer_labels



def pad_with_max_val(gt_list, val):
    """if the number of elements in gt is less than MAX_DECODE_STEPS, we pad it with the max value in a class"""
    if len(gt_list) < gv.MAX_DECODE_STEPS:
        gt_list = (
            gt_list
            + (
                np.ones(
                    gv.MAX_DECODE_STEPS - len(gt_list),
                )
                * val
            ).tolist()
        )
    return gt_list


def str_replace(ans):
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    return ans


def str_replace_(info, ans_opt):
    ans = info[ans_opt]
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    ans = ans.replace("Impossible", "0")
    info[ans_opt] = ans
    return ans


def get_val(qinfo, ans_opt, is_one_of_option=False):
    """get the value of the answer option. This code also encodes the value into a number by removing extreneous strings"""
    """ is_one_of_option is True, when ans_opt is one of the options, need not be the correct answer option."""
    where = lambda x, y: np.where(np.array(x) == y)[0][0]

    pid = int(qinfo["puzzle_id"])
    if pid in [16, 18, 35, 39, 63, 100]:
        ans = qinfo[ans_opt]
        if pid == 16:
            ans_opt_val = [int(ii) for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 18:
            ans_opt_val = [int(ii) for ii in ans.split("-")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 35:
            ans_opt_val = [
                ord(ii) - ord("A") for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")
            ]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 39:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 63:
            ans_opt_val = [
                int(ii)
                for ii in ans.replace("and", ",")
                .replace("or", ",")
                .replace(", ,", ",")
                .replace("only", "")
                .replace(" ", "")
                .split(",")
            ]
            key = str(63)
            if key in gv.NUM_CLASSES_PER_PUZZLE:
                ans_opt_val = pad_with_max_val(ans_opt_val, gv.NUM_CLASSES_PER_PUZZLE[key] - 1)
        elif pid == 100:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        ans_opt_val = np.array(ans_opt_val)

    elif pid == 58:
        # puzzle 58 has answers as <operator><one digit number>, e.g./4,-5, etc.
        # we use +=1, -=2, x=3, /=4. so /4 will be 44, -5=25, +2= 2.
        ans_opt_val = qinfo[ans_opt]
        ans_opt_val = (where(gv.signs, ans_opt_val[0]) + 1) * 10 + int(ans_opt_val[1:])
    elif pid == 25:
        # we need to fix the time in AM/PM format properly.
        ans = qinfo[ans_opt]
        ans_opt_val = int(ans.replace(":00 AM", "").replace(":00 PM", ""))
        if ans.find("PM") > -1:
            ans_opt_val += 12
    else:
        try:
            ans_opt_val = int(qinfo[ans_opt])
        except:
            if len(qinfo[ans_opt]) > 0:
                try:
                    ans_opt_val = ord(qinfo[ans_opt]) - ord("A")
                except:
                    try:
                        ans_opt_val = str_replace(qinfo[ans_opt])
                        ans_opt_val = ans_opt_val.replace("Impossible", "0")  # puzzle 58.
                        if int(qinfo["puzzle_id"]) == 1:  # if the puzzle id is 1, then the options are icon classes.
                            ans_opt_val = "_".join(ans_opt_val.split(" "))
                            if ans_opt_val in gv.icon_class_ids:
                                ans_opt_val = where(gv.icon_class_ids, ans_opt_val)
                            elif ans_opt_val + "s" in gv.icon_class_ids:
                                ans_opt_val = where(gv.icon_class_ids, ans_opt_val + "s")
                        ans_opt_val = int(ans_opt_val)
                    except:
                        print(qinfo)
                        pdb.set_trace()
            else:
                ans_opt_val = ord(ans_opt) - ord("A")
    if not is_one_of_option:  # implies we are encoding the correct answer.
        qinfo["AnswerValue"] = ans_opt_val
    return ans_opt_val


def get_puzzle_class_info(args):
    #    global SEQ_PUZZLES, puzzle_diff_str, puzzle_diff
    puzzle_classes = {}
    for puzzle_id in args.puzzle_ids:
        puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[args.train_diff] + "/"
        csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[args.train_diff])
        qa_info = read_csv(os.path.join(args.data_root, puzzle_root, csv_file), puzzle_id)

        pid = int(puzzle_id)
        if pid not in gv.SEQ_PUZZLES:
            num_classes = np.array([get_val(qa, qa["Answer"]) for qa in qa_info]).max() + 1
        else:
            if pid in [16, 39, 100]:
                num_classes = 26 + 1  # if the output is a string of numbers, and the max classes is - max val.
            elif pid in [18, 35]:
                num_classes = 5 + 1  # the minus one is for end of items.
            elif pid in [63]:
                num_classes = np.array([get_val(qa, qa["Answer"]).max() for qa in qa_info]).max() + 1
        puzzle_classes[str(puzzle_id)] = num_classes
    return puzzle_classes


def fix_acc(acc_list):
    """removes accuracy for puzzles in gv.puzzles_not_included"""
    idx = np.array(list(set(np.arange(1, gv.num_puzzles + 1)).difference(set(gv.puzzles_not_included))))
    new_acc_list = acc_list[idx - 1]
    return new_acc_list

def read_dataset_info(csvfilename):
    import csv

    qa_info = {}
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            key = str(row["type"]).lower()
            if key not in qa_info.keys():
                qa_info[key] = [row["puzzle_id"]]
            else:
                qa_info[key].append(row["puzzle_id"])
    assert np.array([len(qa_info[key]) for key in qa_info.keys()]).sum() == 101
    return qa_info


class NoWarningFilter(logging.Filter):
    def filter(self, record):
        target = "max_new_tokens` (=2) and `max_length`(=51) seem to have been set. `max_new_tokens` will take precedence."
        if target in record.msg:
            return False
        else:
            return True

def set_save_dir(model_args, data_args, training_args):
    """_summary_
        modify output_dir to be '{training_args.output_dir}/{model_args.model_type}/{data_args.prediction_type}/{run_name}'
        For diverse experiments
        + 만약 저장하는 option이 없다 -> 그냥 return하기 
    Args:
        model_args (_type_): _description_
        data_args (_type_): _description_
        training_args (_type_): _description_

    Returns:
        _type_: _description_
    """
    if training_args.save_strategy == "no":
        return training_args

    modified_output_dir = ""
    if training_args.run_name == None:
        raise ValueError("Training argument 'run_name' is not defined, but 'save_strategy' argument defined. Checkpoints \
                            should save under '(training_args.output_dir)/(model_args.model_type)/(data_args.prediction_type)/(run_name)/' \
                            please set 'run_name' argument from your shell script")
    else:
        modified_output_dir = os.path.join(training_args.output_dir, model_args.model_type, data_args.prediction_type, training_args.run_name)
        training_args.output_dir = modified_output_dir

    return training_args


def is_float(value):
    """
        value을 '3.0', '3', 'x 3', 'PM 16:00' 이런식으로 다양하게 나오기에 채점하기 알맞게 바꿔줘야함
        기본적으로 int/float가 아닌 경우 embedding유사한 걸로 근사
        주의) 
        1. '+ 3' 같은 경우 float, int처리하면 그냥 3으로 나와버림 
        2. '- 3' 같이 연산을 답으로 하는 경우가 있는 반면, '-3'같은 그냥 음수의 숫자가 답인 경우가 있는데 이를 구분해야 함

    Args:
        value (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        # Try converting to an integer
        num = float(value)
        return True
    except ValueError:
        return False


def cosine_similarity(vector1, vector2):
    # Convert lists to NumPy arrays for efficient calculations
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Compute dot product
    dot_product = np.dot(vector1, vector2)
    
    # Compute magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity


def bertemb_test():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence = ['-7', '/7','/4','/1','/9']
    target = ['-8']

    embeddings = model.encode(sentence)
    target_embedding = model.encode(target)


    breakpoint()
    
    




# if __name__ == '__main__':
#     # value = 3.0
#     # print(convert_to_number(value))
#     bertemb_test()