import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
import glob

def generate_caption(split):
    base_directory = f'/home/work/g-earth-22/VLM/VLM/database/ScienceQA/captions/{split}'
    json_files = glob.glob(os.path.join(base_directory, '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)

    return caption_dict

def load_data_std():
    problems = json.load(open(os.path.join(data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(data_root, 'pid_splits.json')))

    train_qids = pid_splits['%s' % ('train')]
    val_qids = pid_splits['%s' % ('val')]
    test_qids = pid_splits['%s' % ('test')]
    # print(f"number of train problems: {len(train_qids)}\n")
    # print(f"number of val problems: {len(val_qids)}\n")
    # print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids

def generate_scienceqa_qainfo(problems, qids, split):
    results = []
    caption_dict = generate_caption(split)
    count = 0
    for i in qids[split]:
        try:
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
                if os.path.isfile(os.path.join(data_root, 'SAM_features', split, f'{i}.npy')):
                    qa_dict['sam_feature_path'] = os.path.join(data_root, 'SAM_features', split, f'{i}.npy')
                    qa_dict['caption'] = caption_dict[str(i)]['caption']
                    results.append(qa_dict)
                else:
                    count += 1
        except:
            count+=1
    print(f'scienceqa / {count}')

    return results

# Task: {'yes or no', 'true-or false', 'closed choice'}
# closed choice만 추출: 12726 -> 12254

data_root = '/home/work/g-earth-22/VLM/VLM/database/ScienceQA'
split = 'train'

problems, qids = load_data_std()
scienceqa_qa_dict = generate_scienceqa_qainfo(problems, qids, split)
print('scienceqa data num :', len(scienceqa_qa_dict))

# print(results[0])
 