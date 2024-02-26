"""
    serial number을 기준으로 
    Input : system message, few shot qa , action annotation (optional), query context
    을 GPT4V 에 보내는 코드
"""
import os
import json
import base64
import requests
import argparse
from tqdm import tqdm
import tiktoken
import math
import re
from urllib import request
from io import BytesIO
import base64
import copy
from PIL import Image
import numpy as np


# Helper functions

def get_image_dims(image):
    # regex to check if image is a URL or base64 string
    url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    if re.match(url_regex, image):
        response = request.urlopen(image)
        image = Image.open(response)
        return image.size
    elif re.match(r'data:image\/\w+;base64', image):
        image = re.sub(r'data:image\/\w+;base64,', '', image)
        image = Image.open(BytesIO(base64.b64decode(image)))
        return image.size
    else:
        raise ValueError("Image must be a URL or base64 string.")

def calculate_image_token_cost(image, detail):
    # Constants
    LOW_DETAIL_COST = 85
    HIGH_DETAIL_COST_PER_TILE = 170
    ADDITIONAL_COST = 85

    if detail == 'low':
        # Low detail images have a fixed cost
        return LOW_DETAIL_COST
    elif detail == 'high':
        # Calculate token cost for high detail images
        width, height = get_image_dims(image)
        # Check if resizing is needed to fit within a 2048 x 2048 square
        if max(width, height) > 2048:
            # Resize the image to fit within a 2048 x 2048 square
            ratio = 2048 / max(width, height)
            width = int(width * ratio)
            height = int(height * ratio)

        # Further scale down to 768px on the shortest side
        if min(width, height) > 768:
            ratio = 768 / min(width, height)
            width = int(width * ratio)
            height = int(height * ratio)
        # Calculate the number of 512px squares
        num_squares = math.ceil(width / 512) * math.ceil(height / 512)

        # Calculate the total token cost
        total_cost = num_squares * HIGH_DETAIL_COST_PER_TILE + ADDITIONAL_COST

        return total_cost
    else:
        # Invalid detail_option
        raise ValueError("Invalid detail_option. Use 'low' or 'high'.")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4-vision-preview" in model:
        num_tokens=0
        for message in messages:
            for key, value in message.items():
                if isinstance(value, list):
                    for item in value:
                        num_tokens += len(encoding.encode(item["type"]))
                        if item["type"] == "text":
                            num_tokens += len(encoding.encode(item["text"]))
                        elif item["type"] == "image_url":
                            num_tokens += calculate_image_token_cost(item["image_url"]["url"], item["image_url"]["detail"])
                elif isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def generate_query_msg(args, manual_num):
    msg = [
        {
            "role" : "user",
            "content" : [

            ]
        }
    ]
    # load image and encode
    img_path = os.path.join(args.manual_path, manual_num, 'low_res_images')
    images = os.listdir(img_path)
    images = sorted(images, key=lambda x: int(x.split('-')[1].split('.')[0]))
    all_images = []
    for image in images:
        with open(os.path.join(img_path,image), "rb") as image_file:
            all_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

    for image_feat in all_images:
        msg[0]["content"].append(
            {
                "type" : "image_url",
                "image_url" : {
                    "url" : f"data:image/jpg;base64,{image_feat}",
                    "detail" : "low"
                }
            }
        )

    return msg



def fewshot_singlecase_annotation(args, manual_num, annot=None):
    """
        manual_num : few shot qa sample json path 
    """
    fewshot_sample_path = os.path.join(args.fewshot_path,manual_num)
    #fewshot serial num
    with open(fewshot_sample_path, 'r') as f:
        fewshotqa = json.load(f)

    question = fewshotqa[f"cat{args.category_type}"]["question"]
    answer = fewshotqa[f"cat{args.category_type}"]["answer"]


    msg = [
        {
            "role": "user",
            "content": [
            ]
        }
    ]
    # load image and encode
    img_path = os.path.join(args.manual_path, manual_num.split('.json')[0], 'low_res_images')
    images = os.listdir(img_path)
    images = sorted(images, key=lambda x: int(x.split('-')[1].split('.')[0]))
    all_images = []
    for image in images:
        with open(os.path.join(img_path,image), "rb") as image_file:
            all_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
    # breakpoint()
    for image_feat in all_images:
        msg[0]["content"].append(
            {
                "type" : "image_url",
                "image_url" : {
                    "url" : f"data:image/jpg;base64,{image_feat}",
                    "detail" : "low"
                }
            }
        )
    
    # optionally append action annotation
    if args.mode == 'w_annot':
        #NOTE : need to add action annotation
        all_vid_caps = annot[manual_num.split('.json')[0]]["preprocessed_video_caption"]
        # 만약 caption 개수가 30개 이상인 경우, 30개로 sampling
        sampled_caption = ""
        if len(all_vid_caps.keys()) > 20:
            #length을 기준으로 linspace -> key 값으로 들어가게끔 한다.. .
            sampled_caption_idx = np.linspace(0, len(all_vid_caps.keys())-1, 20, dtype=int)
            for idx, slice_idx in enumerate(sampled_caption_idx):
                single_cap = f"{idx+1}. "+all_vid_caps[f"verb_caption_{slice_idx}"] + '.\n'
                sampled_caption += single_cap
        else:
            cnt = 1
            for k, v in all_vid_caps.items():
                sampled_caption += f"{cnt}. " + all_vid_caps[k] + '.\n'
                cnt+=1
        msg[0]["content"].append(
            {
                "type" : "text",
                "text" : sampled_caption
            }
        )


    #add question and answer
    msg.append({
        "role" : "assistant",
        "content" : [

        ]
    })
    #add question
    msg[1]["content"].append(
        {
            "type" : "text",
            "text" : f"Question :\n{question}\n===\n Answer : \n{answer}\n"
        }
    )

    return msg


def generate_system_msg(category_type, sys_msg_path, mode):
    with open(sys_msg_path, 'r') as f:
        sys_candidate = json.load(f)
    common_prompt = sys_candidate[f"{mode}_common"]
    category_prompt = sys_candidate[f"cat{category_type}"]
    input_prompt = common_prompt + category_prompt
    msg = [
        {
        "role": "system",
        "content": [
            {
            "type": "text",  
            "text": f"{input_prompt}"
            },
        ]
        }
    ]


    return msg



def send_request(api_key, payload):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


def evaluate_response(result_cache, error_cache, query_serial_num, response):
    """
        if response is error status -> update error file
        else -> update response file
    """
    # if error, save to error case
    if 'error' in response.json().keys():
        error_cache[query_serial_num] = response.json()
    else:
        result_cache[query_serial_num] = response.json()

    return result_cache, error_cache








def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category_type', type=str, default="1", help='question cateogry type (1~6)')
    parser.add_argument('--fewshot_path', type=str, required=True, help='fewshot examples folder path')
    parser.add_argument('--sys_msg_path', type=str, required=True, help='system message json path')
    parser.add_argument('--mode', type=str, required=True, help='without annotation or with annotation (wo_annot / w_annot)')
    parser.add_argument('--single_turn', action='store_true', help="check whether it's single turn or multiturn generation")
    parser.add_argument('--api_mapping_path', type=str, required=True, help='mapping json for api keys and manuals')
    parser.add_argument('--api_num', type=str, required=True, help='what api key to use')
    parser.add_argument('--annot_path', type=str, help="action annotation json path")
    parser.add_argument('--manual_path', type=str, help="path to retrieve images of manual")
    parser.add_argument('--query_list', type=str, help="use for query candidates. Key values are query id")
    parser.add_argument('--save_path', type=str, help="generated result path")

    return parser.parse_args()


def main(args):
    print(f"QA generation {args.mode}, save path is {args.save_path}")
    # get api key and api num from argument
    with open(args.api_mapping_path, 'r') as f:
        ingredients = json.load(f)
    
    api_key = ingredients[f"API_{args.api_num}"]["api"]
    query_list = ingredients[f"API_{args.api_num}"]["manuals"] #list

    # 결과, 에러 cache file 미리 저장 -> 추후에 읽을것임
    response_result = {}
    error_case_handling = {}
    with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_{args.api_num}.json'), 'w') as f:
        json.dump(response_result, f)
    with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_error_{args.api_num}.json'), 'w') as f:
        json.dump(error_case_handling, f)
    

    #create dict for save response and error cases
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [

        ],
        "max_tokens": 1024
    }
    if args.annot_path:
        with open(args.annot_path, 'r') as f:
            annot = json.load(f)
    else:
        annot = None

    #=========================Add system message ================================
    sys_msg = generate_system_msg(args.category_type, args.sys_msg_path, args.mode)
    for msg in sys_msg:
        payload["messages"].append(msg)

    #=========================Add few shot sample ================================
    #get few shot sample's serial number
    fewshot_serials = os.listdir(args.fewshot_path)
    for sample in fewshot_serials:
        fewshot_msg = fewshot_singlecase_annotation(args, sample, annot)
        for msg in fewshot_msg:
            payload["messages"].append(msg)

    # print(f"{num_tokens_from_messages(payload['messages'], 'gpt-4-vision-preview')} token in common prompt")
    # end building common payload 

    #===========================Add each query sample===========================
    #finally add query context
    for each_query in tqdm(query_list):
        try:
            with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_{args.api_num}.json'), 'r') as f:
                response_result = json.load(f)
            with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_error_{args.api_num}.json'), 'r') as f:
                error_case_handling = json.load(f)

            final_payload = copy.deepcopy(payload)
            query_msg = generate_query_msg(args,each_query)
            for msg in query_msg:
                final_payload["messages"].append(msg)
            # print(f"{num_tokens_from_messages(final_payload['messages'], 'gpt-4-vision-preview')} single payload")

            #========================send request / save response =======================
            response = send_request(api_key, final_payload)
            response_result, error_case_handling = evaluate_response(response_result,error_case_handling, each_query, response)

            with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_{args.api_num}.json'), 'w') as f:
                json.dump(response_result, f)
            with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_error_{args.api_num}.json'), 'w') as f:
                json.dump(error_case_handling, f)
        
        except Exception as e:
            print(e)
            print(f"other error case : {each_query}")

    return



if __name__ == '__main__':
    args = parse_args()
    main(args)
