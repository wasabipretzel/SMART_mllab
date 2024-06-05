import pandas as pd
import json
import os

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

data_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVision'
filename = '/home/work/g-earth-22/VLM/VLM/database/MathVision/data/test.jsonl'
df = load_jsonl_with_pandas(filename)

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
    
mathvision_qa_dict = generate_mathvision_qainfo(df)
print('mathvision data num :', len(mathvision_qa_dict))

# print (len(mathvision_qa_dict))
# print (mathvision_qa_dict[0])
# print (mathvision_qa_dict[100])