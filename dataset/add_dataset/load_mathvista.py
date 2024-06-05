import pandas as pd
import json
import os


data_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVista'
img_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVista/'
filename = '/home/work/g-earth-22/VLM/VLM/database/MathVista/data/testmini.json'
hint_filename = '/home/work/g-earth-22/VLM/VLM/database/MathVista/data/query.json'

df = pd.read_json(filename)
with open(hint_filename, 'r', encoding='utf-8') as file:
    df_hint = json.load(file)

def generate_mathvista_qainfo(df):
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
    

mathvista_qa_dict = generate_mathvista_qainfo(df)
print('mathvista data num :', len(mathvista_qa_dict))

# print (len(mathvista_qa_dict))
# print (mathvista_qa_dict[0])
# print (mathvista_qa_dict[1])
# print (mathvista_qa_dict[2])
# print (mathvista_qa_dict[100])