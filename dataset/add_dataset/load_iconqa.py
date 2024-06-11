import os
import json
import glob

data_path = '/home/work/g-earth-22/VLM/database/ORIGINAL/IconQA/iconqa_data'
split_list = ['train', 'val', 'test']

def generate_caption(split):
    base_directory = f'/home/work/g-earth-22/VLM/database/ORIGINAL/IconQA/iconqa_data/iconqa/captions/{split}/choose_txt'
    json_files = glob.glob(os.path.join(base_directory, '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)

    return caption_dict

def generate_iconqa_qainfo(data_path, split_list):
    qa_dict = []
    with open(os.path.join(data_path, 'problems.json'), "r") as file: 
        problems = json.load(file)
    with open(os.path.join(data_path, 'pid_splits.json'), "r") as file: 
        pid_splits = json.load(file)


    for split in split_list:
        caption_dict = generate_caption(split)
        mc_pids = pid_splits[f'choose_txt_{split}'] # train+test+val 31578개
        opt_char = ['A', 'B', 'C', 'D', 'E']
        count = 0
        for pid in mc_pids:
            try:
                single_dict = {}
                cur_problem = problems[pid]
                
                single_dict['Question'] = cur_problem['question']
                single_dict['image'] = os.path.join(data_path, f'iconqa/{split}/choose_txt/{pid}/image.png')
                for i, opt in enumerate(cur_problem['choices']):
                    single_dict[opt_char[i]] = opt
                    if i == cur_problem['answer']:
                        single_dict['Answer'] = opt_char[i]
                        single_dict['AnswerValue'] = opt

                if os.path.isfile(os.path.join(data_path, 'iconqa/SAM_features', f'{split}/choose_txt/{pid}.npy')):
                    single_dict['sam_feature_path'] = os.path.join(data_path, 'iconqa/SAM_features', f'{split}/choose_txt/{pid}.npy')
                    single_dict['caption'] = caption_dict[str(pid)]['caption']
                    qa_dict.append(single_dict)
                else:
                    count += 1
            except:
                count += 1
        
        print(f'iconqa / {split} / {count}')

        # print(f"Split : {split}, total_num_data : {len(qa_dict)}")
    return qa_dict

iconqa_qa_dict = generate_iconqa_qainfo(data_path, split_list)
print('iconqa data num :', len(iconqa_qa_dict))