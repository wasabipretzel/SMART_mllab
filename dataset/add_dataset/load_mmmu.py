import pandas as pd
import os
import glob
import base64
import io
from PIL import Image
import json


def load_all_files(base_directory):
    file_paths = glob.glob(os.path.join(base_directory, '**', '*'), recursive=True)
    
    files = []
    for f in file_paths:
        if os.path.isfile(f) == True:
            if 'SAM_features' not in f:
                if 'README.md' not in f:
                    if 'test' not in f:
                        if 'caption' not in f:
                            files.append(f)
    return files

def generate_caption(): # todo
    base_directory = '/home/work/g-earth-22/VLM/VLM/database/MMMU'
    json_files = glob.glob(os.path.join(base_directory, 'captions', '*', '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)
            # print('len', len(caption_dict))

    return caption_dict

base_path = '/home/work/g-earth-22/VLM/VLM/database/MMMU/'
files = load_all_files(base_path)

def load_image(image_data: str):
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def count_num_img(datum):
    n_img = 0

    for i in range(7):
        key = 'image_%s'%(i+1)
        if datum[key] != None:
            n_img += 1
    return n_img

def generate_mmmu_qainfo(df):
    qtype_count = 0
    mc_qtype_count = 0
    max_img_count = 0
    max_opt_count = 0
    # option = f"{file.split(base_path)[1].split('/')[0]}"
    caption_dict = generate_caption()
    opt_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    qa_dict = []
    num_data = len(df)
    count = 0

    for i in range(num_data):
        try:
            datum = df.iloc[i]

            if datum['question_type'] != 'multiple-choice':
                qtype_count += 1
                continue

            else: # multiple-choice samples: 
                mc_qtype_count += 1

                single_dict = {}
                single_dict['id'] = datum['id']

                n_img = count_num_img(datum)
                if n_img != 1:
                    max_img_count += 1
                    continue
                else:
                    image = load_image(datum['image_1']['bytes'])
                    single_dict['image'] = image

                single_dict['Question'] = datum['question'].strip()
                options = eval(datum['options'])

                try: 
                    for j, opt in enumerate(options):
                        single_dict[opt_char[j]] = opt
                except:
                    max_opt_count += 1
                    continue

                single_dict['Answer'] = datum['answer']
                single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
                cur_option = '_'.join(datum['id'].split('_')[1:-1])
                cur_idx = datum['id'].split('_')[-1]
                if os.path.isfile(os.path.join(base_path, 'SAM_features', cur_option, f'{cur_idx}.npy')):
                    single_dict['sam_feature_path'] = os.path.join(base_path, 'SAM_features', cur_option, f'{cur_idx}.npy') # todo
                    single_dict['caption'] = caption_dict[datum['id']]['caption']
                    qa_dict.append(single_dict)
                else:
                    count+=1
        except:
            count+=1
    print(f'mmmu / {count}')

    # print (qtype_count, mc_qtype_count, max_img_count, max_opt_count) 
    return qa_dict


df = pd.concat(
    pd.read_parquet(file)
    for file in files
)

mmmu_qa_dict = generate_mmmu_qainfo(df)

# for k, file in enumerate(files):
#     df = pd.read_parquet(file)
#     if k == 1:
#         mmmu_qa_dict = generate_mmmu_qainfo(df, file)
#     else:
#         mmmu_qa_dict += generate_mmmu_qainfo(df, file)
print('mmmu data num :', len(mmmu_qa_dict))
# print (len(mmmu_qa_dict))
# print (mmmu_qa_dict[0])
# print (mmmu_qa_dict[1])
# print (mmmu_qa_dict[2])
# print (mmmu_qa_dict[100])