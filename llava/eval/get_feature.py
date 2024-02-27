import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os
import numpy as np
from tqdm import tqdm


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def feature_extract(args):
    """
        1. Get target serial numbers
        2. for each serial, retrieve images, generate save folder, extract feature
            make to [I(num images), D] tensor. 
        3. Save it.
    """
    # Model
    disable_torch_init()



    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    image_processor.do_center_crop = False

    vision_encoder = model.model.vision_tower
    
    #get target serial number 
    serial_list = os.listdir(args.image_folder)

    for each_serial in tqdm(serial_list):
        images_path = os.path.join(args.image_folder, each_serial, 'images')
        all_images = os.listdir(images_path)
        all_images = sorted(all_images, key=lambda x: int(x.split('-')[1].split('.')[0]))
        result_tensor = []
        for each_img in all_images:
            each_img_path = os.path.join(images_path, each_img)
            each_image = load_images([each_img_path])
            img_tensor = process_images(
                each_image,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16) #[B, C, H, W]
            feat = vision_encoder(img_tensor).squeeze() #[num_tokens, D(1024)]
            result_tensor.append(feat.cpu().detach().numpy())
        result_feat = np.stack(result_tensor, axis=0)  #[NUM_Image, 576, D]
        
        np.save(os.path.join(args.output_folder, each_serial+'.npy'), result_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    feature_extract(args)


    #input folder (images), output folder (llava_ft_feature)
    # page1~.. 
