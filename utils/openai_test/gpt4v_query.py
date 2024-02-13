import base64
import os
import requests


# Function to encode the image
def encode_image(image_path):
    #with multiple images -> return with listed values
    sev_images = os.listdir(image_path)
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
    "max_tokens": 256
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






def main(api_key, image_path, general_prompt, question):

    base64_images = preprocess_image(image_path)

    payload = payload_generator(general_prompt, question, base64_images)

    response = send_request(api_key, payload)
    
    print(response.json())
    #TODO : need to save response results



if __name__ == '__main__':
    # OpenAI API Key
    api_key = "" #NOTE need to add your own openai key
    # general_prompt = "The following images are instructional images to assist with assembling furniture. Page numbers are in the bottom left or bottom right. The bold numbers displayed in the upper left corner are the assembly order."
    general_prompt = "The following images are instructional images to assist with assembling furniture. Page numbers are in the bottom left or bottom right. The bold numbers displayed in the upper left corner are the assembly order."
    question = "Question : What happened to the furniture during step 2 (page 12) and 3 (page 12)? What is changed?"
    # image_path = "/openai_test/test"
    image_path = "/openai_test/playground"

    main(api_key, image_path, general_prompt, question)
