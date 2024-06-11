import base64
import json
import io
import os
import math
import pandas as pd
from PIL import Image
import glob

data_dir = '/home/work/g-earth-22/VLM/VLM/database/MMStar'
filename = '/home/work/g-earth-22/VLM/VLM/database/MMStar/MMStar.tsv'
df = pd.read_csv(filename, sep='\t')

def generate_caption():
    base_directory = f'/home/work/g-earth-22/VLM/VLM/database/MMStar/captions'
    json_files = glob.glob(os.path.join(base_directory, '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)

    return caption_dict

def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = load_image(io.BytesIO(image_data))
    return image

def generate_mmstar_qainfo(df):
    count = 0
    invalid_opt = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']
    caption_dict = generate_caption()
    qa_dict = []
    num_data = len(df)
    count = 0
    # refine invalid_ques and opts
    for i in range(num_data): # 1500 samples
        try:
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
            if os.path.isfile(os.path.join(data_dir, 'SAM_features', f'{i}.npy')):
                single_dict['sam_feature_path'] = os.path.join(data_dir, 'SAM_features', f'{i}.npy') # todo
                single_dict['caption'] = caption_dict[str(i)]['caption']
                qa_dict.append(single_dict)
            else:
                count+=1
        except:
            count+=1
    print(f'mmstar / {count}')

    return qa_dict
    
mmstar_qa_dict = generate_mmstar_qainfo(df)
print('mmstar data num :', len(mmstar_qa_dict))
# 
# print (len(mmstar_qa_dict))
# print (mmstar_qa_dict[0])
# print (mmstar_qa_dict[1])
# print (mmstar_qa_dict[2])
# print (mmstar_qa_dict[100])