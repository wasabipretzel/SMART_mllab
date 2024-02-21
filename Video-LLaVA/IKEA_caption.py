import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import os
import re
import json
from tqdm import tqdm

def main():
    disable_torch_init()
    inp = 'This is a video clip of part of the assemblying furniture. Please describe what action is being performed in this clip with few sentences.'
    model_path = '/SeqMMLearning/Video-LLaVA/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda:2'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # print(prompt)
    

    with open('/data/dataset/mapping/vid_valid_mapping.json', 'r') as f:
        vid_mapping = json.load(f)
    video_base_path = "/data/dataset/videos"
    total_caption_dict = {}
    serials = list(vid_mapping.keys())
    # vid_list = "/data/IKEA/dataset/downscaled_videos/m7KIMoQMap4.mp4'
    for data_id in tqdm(serials, desc="Caption Inference..."):
        # try:
        video_metadata = vid_mapping[data_id]
        print(data_id)
        video_caption = []
        #path
        video = os.path.join(video_base_path, video_metadata["vid_name"]+'.mp4')
        # video = "/data/IKEA/dataset/downscaled_videos/m7KIMoQMap4.mp4'
        
        print(f'"{video}"')
        
        video_tensor_list = video_processor(video, return_tensors='pt', **video_metadata)['pixel_values']
        # print(video_tensor_list)
        # assert True==False
        
        breakpoint()
        for i, video_tensor in enumerate(video_tensor_list):
            if type(video_tensor) is list:
                tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
            else:
                tensor = video_tensor.to(model.device, dtype=torch.float16)
                tensor = tensor.unsqueeze(0)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # print(outputs,'\n')
            breakpoint()
            if "In this video clip," in outputs:
                outputs = outputs.replace("In this video clip, ", "")
                outputs = outputs.replace("</s>", "")
            else:
                outputs = outputs.replace("</s>", "")
            
            video_cap_dict = {"video_caption_"+ str(i): outputs}
            video_caption.append(video_cap_dict)

        total_caption_dict[data_id] = video_caption
        breakpoint()
        
            
        # except Exception as e:
        #     data_id["video_caption"] = []
        #     pass
            
    with open('/SeqMMLearning/Video-LLaVA/generated.json', 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    main()