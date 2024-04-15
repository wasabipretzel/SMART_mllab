import os
import math
import pickle as pkl

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import nltk
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
import transformers
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from utils.util import *

from utils.util import concat_text_input_output, generation_concat_text_input_output

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class SMART_starter(Dataset):
    """
        single image를 읽고 processing해서 올린다 (instructblip processor어디서 동작하는지 찾아볼것)
        question and answer 밀어올리는 방식은 
    """
    def __init__(self, data_args, mode, processor=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data_args = data_args
        self.mode = mode
        # if processor != None:
        #     self.processor = processor
        self.qa_info = self.get_qainfo()
        """
            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
        self.transform = processor
        self.no_question = False
        self.SEQ_PUZZLES=[16, 18, 35, 39, 63, 100]
        self.max_qlen = 110
        self.max_olen=4

        self.vocab_source_path = "/SMART_mllab/etc/data/v2/vocab_init_source.pickle"
        with open(self.vocab_source_path, 'rb') as f:
            vocab_source = pkl.load(f)
        self.vocab = Vocabulary()
        self.vocab.word2idx = vocab_source["word2idx"]
        self.vocab.idx2word = vocab_source["idx2word"]
        self.vocab.idx = vocab_source["idx"]

    def quest_encode(self, question):
        tokens = nltk.tokenize.word_tokenize(question.lower())
        q_enc = np.zeros((self.max_qlen,), dtype="long")
        if not self.no_question:
            enc_tokens = (
                [self.vocab("<start>")] + [self.vocab(tokens[t]) for t in range(len(tokens))] + [self.vocab("<end>")]
            )
            q_enc[: min(self.max_qlen, len(enc_tokens))] = np.array(enc_tokens)
        return q_enc

    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def opts_encode(self, opts, key):
        opts = opts.lower()
        tokens = nltk.tokenize.word_tokenize(opts)
        enc_tokens = [self.vocab(tokens[t]) for t in range(len(tokens))]
        opt_enc = np.zeros((self.max_olen,), dtype="long")
        opt_enc[: min(self.max_olen, len(enc_tokens))] = np.array(enc_tokens)
        return opt_enc

        
    def get_qainfo(self) -> List[dict]:
        """
            load all QA pair & image metadata
        """
        data_path = os.path.join(self.data_args.split_path, self.data_args.split_type, self.mode+'.pkl')
        with open(data_path, 'rb') as f:
            qa_info = pkl.load(f)
        return qa_info

    def load_image(self, qa_pair):
        """
            qa_pair -> 
            {'id': '1', 
            'Question': 'How many ways are there for the feline to reach the bird if the feline can only move horizontally or vertically towards the bird in the grid?', 
            'image': 'puzzle_19_e_1.png', 
            'A': '6', 'B': '13', 'C': '12', 'D': '10', 'E': '9', 
            'Answer': 'D', 'Note': 'C(5|2)', 
            'puzzle_id': '19', 'AnswerValue': 10}
        """
        image_path = os.path.join(self.data_args.data_path, qa_pair["puzzle_id"], "img", qa_pair["image"])
        image = Image.open(image_path).convert("RGB")

        return image


    def __len__(self):
        return len(self.qa_info)
        
    def __getitem__(self, idx):
        single_qa_pair = self.qa_info[idx]
        pid = single_qa_pair["puzzle_id"]
        image = self.transform(self.load_image(single_qa_pair))

        question = self.quest_encode(single_qa_pair["Question"])
        if self.mode == "train":
            opts = 0
        else:
            _ = [str_replace_(single_qa_pair, key) for key in ["A", "B", "C", "D", "E"]]
            opts = [get_val(single_qa_pair, key, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
        answer_label = self.ans_encode(single_qa_pair["Answer"])
        answer_value = single_qa_pair["AnswerValue"]
        answer = np.zeros(
            10, #gv.MAX_DECODE_STEPS
        )
        if int(pid) not in self.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value


        

        data = {
            'image' : image,
            'pid' : torch.tensor(int(pid)),
            # llm input
            'question' : torch.tensor(question), #prompt + "Question :" + question + "Answer : "
            'option': opts,
            # lbl
            'answer_label' : torch.tensor(answer_label),
            'mode' : self.mode, 
            'answer' : torch.tensor(answer),
        }

        return data

@dataclass
class SMART_starter_collator(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        b_images = []
        b_pids = []
        b_questions = []
        b_options = []
        b_answer_labels = []
        b_answer = []
        mode = instances[0]["mode"]

        for each_batch in instances:
            b_images.append(each_batch["image"]) 
            b_pids.append(each_batch["pid"])
            b_questions.append(each_batch["question"])
            b_options.append(each_batch["option"])
            b_answer_labels.append(each_batch["answer_label"])
            b_answer.append(each_batch["answer"])

        b_images = torch.stack(b_images)
        b_pids = torch.stack(b_pids)
        b_questions = torch.stack(b_questions)
        b_answer_labels = torch.stack(b_answer_labels)
        b_answer = torch.stack(b_answer)

        if mode == "train":
            inputs = {
                "images" : b_images,
                "pids" : b_pids,
                "questions" : b_questions,
                "options" : b_options,
                "answer_labels" : b_answer_labels,
                "answers" : b_answer,
                #for eval
                "labels" : b_answer,
                "input_ids" : b_options,
            }
        else:
            # breakpoint()
            b_options = torch.stack([torch.tensor(opt) for opt in b_options])
            inputs = {
                "images" : b_images,
                "pids" : b_pids,
                "questions" : b_questions,
                "answer_labels" : b_answer_labels,  #labels
                "answers" : b_answer, #value
                #for eval
                "labels" : b_answer,
                "input_ids" : [b_options, b_answer_labels, b_pids], #b_options : [B, 5] , b_answer_labels : [B]
            }
        
        return inputs

