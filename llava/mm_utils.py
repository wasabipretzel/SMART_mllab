from PIL import Image
from io import BytesIO
import base64
import re
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # # # '<image>' 기준으로 prompt를 general prompt / QA set으로 분리
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    # def insert_separator(X, sep): # 토큰 사이 SEP token [0] 삽입
    #     return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    # input_ids = []
    # offset = 0
    # if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
    #     offset = 1
    #     input_ids.append(prompt_chunks[0][0])

    # for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
    #     input_ids.extend(x[offset:])

    # if return_tensors is not None:
    #     if return_tensors == 'pt':
    #         return torch.tensor(input_ids, dtype=torch.long)
    #     raise ValueError(f'Unsupported tensor type: {return_tensors}')
    
    # t=3
    #########################################################################################################################################
    if "Question" not in prompt:
        # QA 생성 시 오류가 발생하여 제대로 QA 생성되지 않은 경우
        input_prompt = tokenizer(prompt).input_ids
        input_ids = torch.tensor(input_prompt, dtype=torch.long)
        return input_ids
        
    else:

        IMAGE_TOKEN_INDEX = -200
        QUESTION_TOKEN_INDEX = -300
        
        split_tag = "<im_start>|<im_end>|###"
        prompt_chunks = re.split(split_tag, prompt)
        prompt_chunks_tokenized = [tokenizer(chunk).input_ids for chunk in prompt_chunks if chunk]  # 비어있지 않은 문자열 조각에 대해 토큰화 수행
 
        IMAGE_TOKEN_INDEX = -200 
        QUESTION_TOKEN_INDEX = -300 
        ANSWER_TOKEN_INDEX = -400
        
        # 각 chunk 앞에 해당하는 구분자 토큰을 삽입
        input_ids = []
        input_stop_index = []
        for i, chunk in enumerate(prompt_chunks_tokenized):
            if i == 0:  # system message
                input_ids.extend(chunk)
                input_stop_index.append(len(chunk))
            elif i == 1:  # learnable query
                input_ids.append(IMAGE_TOKEN_INDEX)
                input_ids.extend(chunk)
                input_stop_index.append(len(chunk))
            elif i == 2:  # question
                input_ids.append(QUESTION_TOKEN_INDEX)
                input_ids.extend(chunk)
                input_stop_index.append(len(chunk))
            else:  # answer
                input_ids.append(ANSWER_TOKEN_INDEX)
                input_ids.extend(chunk)
                input_stop_index.append(len(chunk))

        t=3
    
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long), input_stop_index
            raise ValueError(f'Unsupported tensor type: {return_tensors}')


        return input_ids, input_stop_index

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)