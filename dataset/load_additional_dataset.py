import pandas as pd
import json
import os
import numpy as np
import base64
import io
import math
from PIL import Image

def generate_iconqa_qainfo(data_path, split_list):
    qa_dict = []
    with open(os.path.join(data_path, 'problems.json'), "r") as file: 
        problems = json.load(file)
    with open(os.path.join(data_path, 'pid_splits.json'), "r") as file: 
        pid_splits = json.load(file)


    for split in split_list:
        mc_pids = pid_splits[f'choose_txt_{split}'] # train+test+val 31578개
        opt_char = ['A', 'B', 'C', 'D', 'E']

        for pid in mc_pids:
            single_dict = {}
            cur_problem = problems[pid]
            
            single_dict['Question'] = cur_problem['question']
            single_dict['image'] = os.path.join(data_path, f'iconqa/{split}/choose_txt/{pid}/image.png')
            for i, opt in enumerate(cur_problem['choices']):
                single_dict[opt_char[i]] = opt
                if i == cur_problem['answer']:
                    single_dict['Answer'] = opt_char[i]
                    single_dict['AnswerValue'] = opt

            qa_dict.append(single_dict)

        # print(f"Split : {split}, total_num_data : {len(qa_dict)}")
    return qa_dict


def generate_mathverse_qainfo(df):
    img_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVerse/images'

    count = 0
    mc_count = 0
    no_ques = 0
    invalid_ques = 0
    invalid_opts = 0
    invalid_ans = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']

    qa_dict = []
    num_data = len(df) # 3940 samples
    for i in range(num_data):
        datum = df.iloc[i]
        if datum['question_type'] != 'multi-choice': # skip free-form samples
            continue
        else: # multiple-choice samples: 2180
            mc_count += 1

            single_dict = {}
            single_dict['id'] = datum['sample_index'] # what is the difference btw sample_index and problem_index?
            single_dict['image'] = os.path.join(img_dir, datum['image'])

            if datum['question'] == '':
                no_ques += 1
                continue # 436 samples

            try:
                question, option_seqs = datum['question'].split('Choices:')
            except:
                try: 
                    question, option_seqs = datum['question'].split('Choice:')
                except:
                    invalid_ques += 1 # 5 samples
                    continue # no explicit options (options are depicted in the given image)
            single_dict['Question'] = question.strip()

            opts_list = option_seqs.strip().split('\n')
            for j, opts in enumerate(opts_list):
                try:
                    key, value = opts.split(':')
                except:
                    try:
                        key, value = opts.split('.')
                    except:
                        invalid_opts += 1
                        continue # 47 samples

                single_dict[key] = value.strip()

            single_dict['Answer'] = datum['answer']
            if datum['answer'] not in single_dict.keys():
                invalid_ans += 1
                continue # 44 samples
            else:
                qa_dict.append(single_dict)

            single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
    # print (mc_count, no_ques, invalid_ques, invalid_opts, invalid_ans) 
    return qa_dict

def load_jsonl_with_pandas(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error: {e}")
    df = pd.DataFrame(data)
    return df

def generate_mathvision_qainfo(df):
    """
    # subimages == 1532, 301, 31, 11, 7, 3, 3
    """
    img_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVision/'

    qa_dict = []
    num_data = len(df)
    for i in range(num_data):
        datum = df.iloc[i]
        if datum['options'] == []: # free-form samples
            continue
        else: # multiple-choice samples
            single_dict = {}
            single_dict['id'] = datum['id']
            single_dict['Question'] = datum['question']
            single_dict['image'] = os.path.join(img_dir, datum['image'])
            single_dict['A'], single_dict['B'], single_dict['C'], single_dict['D'], single_dict['E'] = datum['options']
            single_dict['Answer'] = datum['answer']
            single_dict['AnswerValue'] = single_dict[datum['answer']] # have to check
            qa_dict.append(single_dict)
    return qa_dict

def load_data_std(data_root):
    problems = json.load(open(os.path.join(data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(data_root, 'pid_splits.json')))

    train_qids = pid_splits['%s' % ('train')]
    val_qids = pid_splits['%s' % ('val')]
    test_qids = pid_splits['%s' % ('test')]

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids

def generate_scienceqa_qainfo(problems, qids, split_list):
    data_root = '/home/work/g-earth-22/VLM/VLM/database/ScienceQA'
    results = []
    for split in split_list:
        for i in qids[split]:
            problem = problems[f'{i}'] 
            if problem['task']=='closed choice' and problem['image'] is not None:
                qa_dict = {}
                qa_dict['id'] = i
                qa_dict['Question'] = problem['lecture'] + ' ' + problem['question']
                qa_dict['image'] = os.path.join(data_root, split, i, problem['image'])
                for num, opt in enumerate(problem['choices']):
                    qa_dict[str(chr(ord('A')+num))] = opt
                qa_dict['Answer'] = str(chr(ord('A')+problem['answer']))
                qa_dict['AnswerValue'] = problem['choices'][problem['answer']]
                results.append(qa_dict)
    return results

def generate_mathvista_qainfo(df):
    img_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVista/'

    count = 0
    mc_count = 0
    max_opt_count = 0
    no_ques = 0
    invalid_ques = 0
    invalid_opts = 0
    invalid_ans = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']

    qa_dict = []
    for i in range(1, 1001): # total 1000 samples
        datum = df[i]

        if datum['metadata']['split'] == 'test':
            continue # no answer

        if datum['question_type'] != 'multi_choice': # skip free-form samples
            count += 1 # 460 samples
            continue

        else: # multiple-choice samples: 540 samples

            if len(datum['choices']) > 5:
                max_opt_count += 1 # 13 samples
                continue

            mc_count += 1

            single_dict = {}
            single_dict['id'] = datum['pid']
            single_dict['image'] = os.path.join(img_dir, datum['image'])

            single_dict['Question'] = datum['question'].strip()
            for j, choice in enumerate(datum['choices']):
                single_dict[opt_char[j]] = choice

            for j, choice in enumerate(datum['choices']):
                if datum['answer'] == choice:
                    single_dict['Answer'] = opt_char[j]

            single_dict['AnswerValue'] = datum['answer']
            qa_dict.append(single_dict)
    
    # print (mc_count, count, max_opt_count) 
    return qa_dict

def load_image(image_path: str):
    return Image.open(image_path).resize((224, 224)).convert("RGB")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = load_image(io.BytesIO(image_data)) # mjkim image를 바로 return해줌
    return image

def generate_mmstar_qainfo(df):
    count = 0
    invalid_opt = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']

    qa_dict = []
    num_data = len(df)
    
    # refine invalid_ques and opts
    for i in range(num_data): # 1500 samples
        datum = df.iloc[i]
    
        single_dict = {}
        single_dict['id'] = datum['index']
        image = decode_base64_to_image(datum["image"])
        single_dict['image'] = image

        try:
            question, options_str = datum['question'].split('Options:')
        except: 
            try:
                question, options_str = datum['question'].split('Choices:')
            except:
                question = datum['question']
                for opt in opt_char:
                    if opt == 'A':
                        question = question.replace('(%s)'%(opt), 'Choices: %s: '%(opt))
                    question = question.replace('(%s)'%(opt), ', %s:'%(opt))
                question, options_str = question.split('Choices:')
        
        single_dict['Question'] = question.strip()

        option_list = options_str.split(', ')
        for option in option_list:
            try:
                key, value = option.split(':')
                single_dict[key.strip()] = value.strip()
            except:
                count += 1

        single_dict['Answer'] = datum['answer']
        try:
            single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
        except:
            for opt in opt_char:
                single_dict[opt] = opt
            single_dict['AnswerValue'] = single_dict[single_dict['Answer']]    

        qa_dict.append(single_dict)
    return qa_dict

def generate_mmbench_qainfo(df):
    no_img = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']

    qa_dict = []
    num_data = len(df)
    
    for i in range(num_data): # 4329 for DEV_EN
        datum = df.iloc[i]
    
        single_dict = {}
        single_dict['id'] = datum['index']
        image = decode_base64_to_image(datum["image"])
        single_dict['image'] = image

        single_dict['Question'] = datum['question'].strip()
        single_dict['A'] = datum['A']
        single_dict['B'] = datum['B']
        single_dict['C'] = datum['C']
        single_dict['D'] = datum['D']

        for opt in opt_char:
            try:
                if math.isnan(single_dict[opt]) == True:
                    del single_dict[opt]
            except:
                continue

        single_dict['Answer'] = datum['answer']
        single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
        qa_dict.append(single_dict)
    return qa_dict