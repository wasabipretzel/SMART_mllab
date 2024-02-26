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




def generate_query_msg(args, manual_num):
    msg = [
        {
            "role" : "user",
            "content" : [

            ]
        }
    ]
    # load image and encode
    img_path = os.path.join(args.manual_path, manual_num, 'images')
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
    img_path = os.path.join(args.manual_path, manual_num.split('.json')[0], 'images')
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
        all_vid_caps = annot[manual_num]["preprocessed_video_caption"]
        # 만약 caption 개수가 30개 이상인 경우, 30개로 sampling
        sampled_caption = ""
        if len(all_vid_caps.keys()) > 30:
            #length을 기준으로 linspace -> key 값으로 들어가게끔 한다.. .
            sampled_caption_idx = np.linspace(0, len(all_vid_caps.keys())-1, dtype=int)
            for idx in sampled_caption_idx:
                single_cap = f"{idx+1}. "+all_vid_caps[f"verb_caption_{idx}"] + '.\n'
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
    parser.add_argument('--api_key', type=str, required=True, help='openai key for generation')
    parser.add_argument('--annot_path', type=str, help="action annotation json path")
    parser.add_argument('--manual_path', type=str, help="path to retrieve images of manual")
    parser.add_argument('--query_list', type=str, help="use for query candidates. Key values are query id")
    parser.add_argument('--save_path', type=str, help="generated result path")

    return parser.parse_args()


def main(args):
    print(f"QA generation {args.mode}, save path is {args.save_path}")
    #create dict for save response and error cases
    response_result = {}
    error_case_handling = {}
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

    
    # end building common payload 

    #===========================Add each query sample===========================
    #finally add query context
    # get query candidates from query folder
    with open(args.query_list, 'r') as f:
        query_data = json.load(f)
    query_list = query_data.keys()
    for each_query in tqdm(query_list):
        try:
            final_payload = payload.copy()
            query_msg = generate_query_msg(args,each_query)
            for msg in query_msg:
                final_payload["messages"].append(msg)

            with open(f'{each_query}_finalpayload.json', 'w') as f:
                json.dump(final_payload, f)
            breakpoint()
        except:
            print("error")
    return
            
    #         #========================send request / save response =======================
    #         response = send_request(args.api_key, final_payload)
    #         response_result, error_case_handling = evaluate_response(response_result,error_case_handling, each_query, response)
    #     except Exception as e:
    #         print(e)
    #         print(f"other error case : {each_query}")
    #     # print(response.json())
    #     # breakpoint(/
    
    # #save result
    # with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}.json'), 'w') as f:
    #     json.dump(response_result, f)
    # with open(os.path.join(args.save_path, f'qa_response_cat{args.category_type}_error.json'), 'w') as f:
    #     json.dump(error_case_handling, f)

    # print(f"Total {len(query_list)} samples, {len(response_result.keys())} generated, {len(error_case_handling.keys())} occured error.")

    # return



if __name__ == '__main__':
    args = parse_args()
    main(args)
