import base64
import os
import requests


# Function to encode the image
def encode_image(image_path):
    #with multiple images -> return with listed values
    sev_images = os.listdir(image_path)
    image_files = []
    #filter readme.md
    for image in sev_images:
        if '.jpg' in image or '.png' in image:
            image_files.append(image)
    #sort images by page num
    sev_images = sorted(image_files, key=lambda x: int(x.split('.')[0]))
    all_images = []
    for image in sev_images:
        with open(os.path.join(image_path,image), "rb") as image_file:
            all_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
    return all_images



def preprocess_image(image_path):

    # Getting the base64 string
    base64_image = encode_image(image_path)
    return base64_image


def payload_generator(general_prompt, question, encoded_images):
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{general_prompt} {question}"
            },
        ]
        }
    ],
    "max_tokens": 512
    }


    for image in encoded_images:
        payload["messages"][0]["content"].append(
            {
                "type" : "image_url",
                "image_url" : {
                    "url" : f"data:image/jpg;base64,{image}",
                    "detail" : "low"
                }
            }
        )

    return payload


def send_request(api_key, payload):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


def payload_generate_with_system_message():
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{general_prompt} {question}"
            },
        ]
        }
    ],
    "max_tokens": 512
    }


    for image in encoded_images:
        payload["messages"][0]["content"].append(
            {
                "type" : "image_url",
                "image_url" : {
                    "url" : f"data:image/jpg;base64,{image}",
                    "detail" : "low"
                }
            }
        )
    return 



def main(api_key, image_path, general_prompt, question):

    base64_images = preprocess_image(image_path)

    payload = payload_generator(general_prompt, question, base64_images)

    response = send_request(api_key, payload)
    
    print(response.json())
    #TODO : need to save response results



if __name__ == '__main__':
    # OpenAI API Key
    api_key = "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" #NOTE need to add your own openai key
    # general_prompt = "The following images are instructional images to assist with assembling furniture. Page numbers are in the bottom left or bottom right. The bold numbers displayed in the upper left corner are the assembly order."
    general_prompt = "The following images are instructional images to assist with assembling furniture. Page numbers are in the bottom left or bottom right. The bold numbers displayed in the upper left corner are the assembly order."
    additional_context_intro = "You can use video action annotation to answer the questions which is provided below. Video action annotation is the sequential annotation of actions recorded in videos based on instructional assembly images.\n" #Since video action annotation is recorded actions of the entire assembly process, so it should be used to understand assembly documents, and only relevant information should be used for inquiries. 
    additional_context = "1. flip table top \n 2. pick up leg \n 3. align leg screw with table thread \n 4. spin leg \n 5. pick up leg \n 6. align leg screw with table thread \n 7. spin leg \n 8. pick up leg \n 9. align leg screw with table thread \n 10. spin leg \n 11. tighten leg \n 12. rotate table \n 13. spin leg \n 14. pick up leg \n 15. align leg screw with table thread \n 16. spin leg \n 17. flip table \n 18. pick up shelf \n 19. attach shelf to table \n"
    question = "Question : Explain detail description of step 3 (page 9)."

    image_path = "/SeqMMLearning/utils/openai_test/playground"

    general_prompt += additional_context_intro
    general_prompt += additional_context
    main(api_key, image_path, general_prompt, question)
