import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import math
import numpy as np
import av
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers
from transformers import VivitImageProcessor





def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len != seg_len:
        end_idx = np.random.randint(converted_len, seg_len)
    else:
        end_idx = converted_len
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices




class MOMA(Dataset):
    def __init__(self, data_args, mode):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data_args = data_args
        self.mode = mode

        self.vids = self.get_vids() #list

        with open(self.data_args.target_path, 'r') as f:
            self.target_json = json.load(f)
        
        self.video_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        

    def get_vids(self):
        """
            get vids for current mode
        """
        with open(self.data_args.split_path, 'r') as f:
            all_vid = json.load(f)
        return all_vid[self.mode]


    def onehot_multilabel(self, idxs, classes):
        one_hot = np.zeros(classes)
        one_hot[idxs] = 1
        return one_hot

    def get_target(self, vid):
        
        targets = self.onehot_multilabel(idxs=self.target_json['vid2idx'][vid], classes=self.data_args.num_class)

        return targets


    def get_video_input(self, vid):
        container = av.open(os.path.join(self.data_args.raw_vid_path, vid+'.mp4'))
        indices = sample_frame_indices(clip_len=self.data_args.sample_frames,
                             frame_sample_rate= container.streams.video[0].frames // self.data_args.sample_frames,
                            seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices) #[T, H, W, C]
        # inputs = self.video_processor(list(video), return_tensors="pt")["pixel_values"] #[1, T, C, H, W]

        inputs = torch.from_numpy(np.array(self.video_processor(list(video))["pixel_values"][0])) #[1, T, C, H, W] -> 이게 훠어어얼씬 빠름

        return inputs


    def __len__(self):
        return len(self.vids)
        
    def __getitem__(self, idx):
        vid_name = self.vids[idx]
        vid_input = self.get_video_input(vid_name) #[1, T, C, H, W]
        target = self.get_target(vid_name) #nparray (91,)
        data = {
            'vid' : vid_name,
            'vid_input' : vid_input,
            'target': torch.tensor(target) #[91] tensor
        }

        return data




@dataclass
class MOMA_collator(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_vid_names = []
        b_vid_inputs = []
        b_target = []

        for each_batch in instances:
            b_vid_names.append(each_batch["vid"])
            b_vid_inputs.append(each_batch["vid_input"].squeeze(0))
            b_target.append(each_batch["target"])

        #target은 stack
        b_vid_inputs = torch.stack(b_vid_inputs) #[B, T, C, H, W]
        b_target = torch.stack(b_target) #[B, class]

        result = {
            "vid_names" : b_vid_names,
            "vid_input" : b_vid_inputs,
            "labels" : b_target,
        }


        return result

