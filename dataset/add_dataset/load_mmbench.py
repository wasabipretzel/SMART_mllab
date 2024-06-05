import base64
import json
import io
import os
import math
import pandas as pd
from PIL import Image

data_dir = '/home/work/g-earth-22/VLM/VLM/database/MMBench'
filename = '/home/work/g-earth-22/VLM/VLM/database/MMBench/MMBench_DEV_EN_legacy.tsv'

df = pd.read_csv(filename, sep='\t')

def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = load_image(io.BytesIO(image_data))
    return image

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
    
mmbench_qa_dict = generate_mmbench_qainfo(df)
print('mmbench data num :', len(mmbench_qa_dict))
# print (len(mmbench_qa_dict))
# print (mmbench_qa_dict[0])
# print (mmbench_qa_dict[1])
# print (mmbench_qa_dict[2])
# print (mmbench_qa_dict[100])