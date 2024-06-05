import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json, glob, random
import os

SPACE=10
data_dir = '/home/work/g-earth-22/VLM/VLM/database/RAVEN'
img_dir = '/home/work/g-earth-22/VLM/VLM/database/RAVEN/images'
filename = '/home/work/g-earth-22/VLM/VLM/database/RAVEN/RAVEN_char_option.json'
df = pd.read_json(filename)

opt_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G' ,'H']
config_list = ['center_single', 'distribute_four', 'distribute_nine', 'left_center_single_right_center_single', \
    'up_center_single_down_center_single', 'in_center_single_out_center_single', 'in_distribute_four_out_center_single']

def get_an_image(img_array, space=SPACE):
    img = Image.fromarray(img_array)
    new_img = Image.new('RGB', (img.width+space*2, img.height+space*2),'white')
    new_img.paste(img, (space, space))
    # img = img.convert('RGB')
    draw = ImageDraw.Draw(new_img)
    draw.rectangle((space,space,img.width+space,img.height+space), outline=(0,0,0), width = 2)

    return new_img

def get_an_option_img(img_array, option_char, space=SPACE):
    fontsize=40
    img = Image.fromarray(img_array)
    new_img = Image.new('RGB', (img.width+space*2, img.height+fontsize+space*2),'white')
    new_img.paste(img, (space, space))

    font = ImageFont.truetype("arial.ttf", fontsize)
    draw = ImageDraw.Draw(new_img)
    draw.rectangle((space,space,img.width+space,img.height+space), outline=(0,0,0), width = 2)
    draw.text((75,img.height+space), text=option_char, fill='black',font=font)

    return new_img

def get_question_mark(space=SPACE):
    question_mark = Image.new('RGB', (160+space*2, 160+space*2),'white')
    draw = ImageDraw.Draw(question_mark)
    font = ImageFont.truetype("arial.ttf", 120)
    draw.text((55,15), text='?', fill='black',font=font)
    return question_mark

def get_concat_h(img1, img2):
    img = Image.new('RGB', (img1.width + img2.width, img1.height),'white')
    img.paste(img1, (0, 0))
    img.paste(img2, (img1.width, 0))

    return img

def get_concat_v(img1, img2,space=0):
    img = Image.new('RGB', (img1.width, img1.height + img2.height+space), 'white')
    img.paste(img1, (0, 0))
    img.paste(img2, (0, space+img1.height))
    return img


def combine_images(quest_array, option_array, new_img_path):
    ## get questions
    quest_imgs = []

    question_mark = get_question_mark()
    for i in range(8):
        quest_imgs.append(get_an_image(quest_array[i]))

    quest_r1 = get_concat_h(get_concat_h(quest_imgs[0], quest_imgs[1]), quest_imgs[2])
    quest_r2 = get_concat_h(get_concat_h(quest_imgs[3], quest_imgs[4]), quest_imgs[5])
    quest_r3 = get_concat_h(get_concat_h(quest_imgs[6], quest_imgs[7]), question_mark)

    quest_img = get_concat_v(get_concat_v(quest_r1, quest_r2),quest_r3)

    ## get options
    option_imgs = []
    for i in range(8):
        option_imgs.append(get_an_option_img(option_array[i],opt_char[i]))

    option_r1 = get_concat_h(get_concat_h(get_concat_h(option_imgs[0], option_imgs[1]), option_imgs[2]), option_imgs[3])
    option_r2 = get_concat_h(get_concat_h(get_concat_h(option_imgs[4], option_imgs[5]), option_imgs[6]), option_imgs[7])

    option_img = get_concat_v(option_r1, option_r2)

    ## resize options
    mywidth = quest_img.size[0]
    wpercent = (mywidth/float(option_img.size[0]))
    hsize = int((float(option_img.size[1])*float(wpercent)))
    option_img = option_img.resize((mywidth,hsize))

    ## combine
    final_img = get_concat_v(quest_img, option_img, space=20)
    final_img.save(new_img_path)
    # final_img.show()

def make_dataset():
    dict_list = []
    for config in config_list:
        old_path = os.path.join('/home/work/g-earth-22/VLM/VLM/database/RAVEN/RAVEN-10000/',config)
        new_path = os.path.join(img_dir,config)

        if not os.path.isdir(new_path):
            os.makedirs(new_path)
            print('Created diretory :' + new_path)

        file_list = glob.glob(old_path+'/*.npz')

        print(len(file_list), old_path)

        for i in range(len(file_list)):
            file_path = file_list[i]
            img_name = file_path.split('/')[-1].split('.')[0]+'.jpg'
            new_img_path = os.path.join(new_path,img_name)
            # print(new_img_path)

            x = np.load(file_path)
            if len(x['image']) != 16: print(new_img_path)
                
            quest_array = x['image'][0:8]
            option_array = x['image'][8:]
            answerindex = int(x['target'])
            answerkey = opt_char[answerindex]
            combine_images(quest_array, option_array, new_img_path)
            
            single_dict = {'img':config+'/'+img_name, 'answer':answerkey }
            dict_list.append(single_dict)
        
        print(len(dict_list))

    with open(data_dir+'/RAVEN_char_option.json','w') as f:
        json.dump(dict_list, f, ensure_ascii=False)


def generate_raven_qainfo(df):

    qa_dict = []
    num_data = len(df) # 3940 samples
    for i in range(num_data):
        datum = df.iloc[i]

        single_dict = {}
        single_dict['id'] = datum['img'].replace('/RAVEN','').replace('.jpg','') # what is the difference btw sample_index and problem_index?
        single_dict['image'] = os.path.join(img_dir, datum['img'])

        question_list = ["Identify the pattern and select the correct image to replace the question mark.",
            "Analyze the given pattern and pick the image that correctly fits where the question mark is.",
            "Observe the pattern and select the correct picture to complete the series.",
            "Can you identify the pattern in the given images and select the image that should replace the question mark?",
            "Can you analyze the pattern in the images and select the correct picture to replace the question mark?",
            "Can you choose the image that should go in place of the question mark?",
            "Could you select the image that belongs where the question mark is?",
            "Can you pick the correct image to replace the question mark?",
            "Would you be able to identify the image that fits in the spot of the question mark?",
            "Can you determine which image should replace the question mark?"
        ]

        single_dict['Question'] = random.choice(question_list)

        opt_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G' ,'H']
        for opt in opt_list:
            single_dict[opt] = opt

        single_dict['Answer'] = datum['answer']
        single_dict['AnswerValue'] = datum['answer']

        qa_dict.append(single_dict)

    return qa_dict
    

# make_dataset()
raven_qa_dict = generate_raven_qainfo(df)
print('raven data num :', len(raven_qa_dict))

# print (len(raven_qa_dict)) # 70000 samples
# print (raven_qa_dict[0])
# print (raven_qa_dict[20000])
# print (raven_qa_dict[50001])