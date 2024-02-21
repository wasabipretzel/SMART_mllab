
import torch
import cv2
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from collections import Counter
decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform

def generate_time_intervals(vid_duration):
    """
        vid_duration : [{'start': 1422.0, 'end': 2433.0, 'action': 0}, {'start': 2454.0, 'end': 3897.0, 'action': 1}, {'start': 4098.0, 'end': 5037.0, 'action': 2},]
        maximum 10 frame당 하나씩 뽑도록 dense한 time interval을 생성 
        return type : list[tuples]
    """
    unit = 80
    time_intervals=[]
    for each_seg in vid_duration:
        start = each_seg["start"]
        end = each_seg["end"]
        duration = each_seg["end"] - each_seg["start"]
        loop = duration // unit + 1
        if duration <= unit:
            time_intervals.append((start, end))
        else:
            for i in range(1, loop+1):
                tmp_end = start + unit
                if tmp_end > end:
                    tmp_end = end
                time_intervals.append((start, tmp_end))
                start = tmp_end

    return time_intervals


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        **kwargs,
):

    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="pyav", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        # end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        end_sec = duration
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        for seg in kwargs["min_vid_duration"]:
            seg["start"] = int(seg["start"] * kwargs["min_vid_fps"])
            seg["end"] = int(seg["end"] * kwargs["min_vid_fps"])
        time_intervals = generate_time_intervals(kwargs["min_vid_duration"])
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        all_frames = []
        for i in range(len(decord_vr)):
            all_frames.append(decord_vr[i])
        frames = decord_vr.get_batch(range(0, len(decord_vr), 1))
        duration = len(decord_vr)
        video_outputs_list = [] 
        for start_frame, end_frame in time_intervals:
            frame_id_list = np.linspace(start_frame, min(end_frame, duration - 1), num_frames, dtype=int)
            # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
            video_data = decord_vr.get_batch(frame_id_list)
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_data = torch.stack(video_data, dim=1)
            video_outputs = transform(video_data)
            video_outputs_list.append(video_outputs)

    elif video_decode_backend == 'opencv':
        # convert segment sec to frame
        for seg in kwargs["vid_annotation"]:
            seg["start"] = int(seg["start"] * kwargs["vid_fps"])
            seg["end"] = int(seg["end"] * kwargs["vid_fps"])
        cv2_vr = cv2.VideoCapture(video_path)

        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cv2_vr.get(cv2.CAP_PROP_FPS)

        # 시간 구간 설정 :
        time_intervals = generate_time_intervals(kwargs["vid_annotation"])
        # time_intervals = [(start, start + 5 * fps) for start in np.arange(0, duration, 5 * fps)]

        video_outputs_list = [] 
        for start_frame, end_frame in time_intervals:
            frame_id_list = np.linspace(start_frame, min(end_frame, duration - 1), num_frames, dtype=int)
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_id_list[0])
            video_data = []
            counter_frame_id_dict = Counter(frame_id_list)
            for idx in range(frame_id_list[0], frame_id_list[-1]+1):
                _, frame = cv2_vr.read()
                if idx in frame_id_list:
                    for _ in range(counter_frame_id_dict[idx]):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
            
            if video_data:  # 비어 있는 경우
                video_data = torch.stack(video_data, dim=1)
                video_outputs = transform(video_data)
                video_outputs_list.append(video_outputs)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs_list

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        config.vision_config.video_decode_backend = 'opencv'
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames, **kwargs) for image in images][0]
            # image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
