import pandas as pd
import json
import os

data_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVerse'
img_dir = '/home/work/g-earth-22/VLM/VLM/database/MathVerse/images'
filename = '/home/work/g-earth-22/VLM/VLM/database/MathVerse/data/testmini.json'
df = pd.read_json(filename)

def generate_mathverse_qainfo(df):
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
    

mathverse_qa_dict = generate_mathverse_qainfo(df)
print('mathverse data num :', len(mathverse_qa_dict))

# print (len(mathverse_qa_dict)) # 1695 samples
# print (mathverse_qa_dict[0])
# print (mathverse_qa_dict[1])
# print (mathverse_qa_dict[2])
# print (mathverse_qa_dict[100])