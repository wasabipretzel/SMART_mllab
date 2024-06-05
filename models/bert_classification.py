"""
    Baseline model of "Are Deep Neural Networks SMARTer than Second Graders?" (R50 + BERT)
"""
import nltk

# make sure nltk works fine.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("downloading nltk/punkt tokenizer")
    nltk.download("punkt")

import argparse
import glob
import os
import pickle
from collections import Counter
import os
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import pickle

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import models


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


class BERT:
    # https://huggingface.co/docs/transformers/model_doc/bert
    def __init__(self):
        super(BERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.model = BertModel.from_pretrained("bert-base-uncased").to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to("cuda")
        outputs = self.model(**inputs)
        word_feats = outputs.pooler_output
        return torch.tensor(word_feats).cuda()


# Vision backbones and language backbones.
class SMART_Net(nn.Module):
    def __init__(self,im_backbone=None):
        super(SMART_Net, self).__init__()
        self.vocab_source_path = "/SMART_mllab/etc/data/v2/vocab_init_source.pickle"
        with open(self.vocab_source_path, 'rb') as f:
            vocab_source = pickle.load(f)
        self.vocab = Vocabulary()
        self.vocab.word2idx = vocab_source["word2idx"]
        self.vocab.idx2word = vocab_source["idx2word"]
        self.vocab.idx = vocab_source["idx"]

        self.num_opts = 5
        self.out_dim = 128  #  64 #
        self.h_sz = 256  # 256 #128 #
        self.dummy_question = None
        self.model_name = "resnet50"
        self.use_clip_text = False
        self.loss_type = "classifier"
        self.monolithic = True
        self.use_single_image_head = True
        self.train_backbone = False
        self.word_embed = "bert"
        # self.sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in args.puzzle_ids]))
        self.max_qlen = 110
        self.word_dim = 768
        self.word_embed_model = BERT()

        if self.loss_type == "classifier":
            # self.max_val = gv.MAX_VAL + 1
            self.max_val = 256+1
            self.loss_fn = nn.CrossEntropyLoss()

        # image backbones.
        # if args.model_name[:6] == "resnet":
        # self.im_feat_size = im_backbone.fc.weight.shape[1]
        self.im_feat_size=2048
        modules = list(im_backbone.children())[:-1]
        self.im_cnn = nn.Sequential(*modules)

        self.create_puzzle_head()

        # language backbones
        if self.word_embed == "standard":
            self.q_emb = nn.Embedding(len(self.vocab), self.h_sz, max_norm=1)
            self.q_lstm = nn.LSTM(self.h_sz, self.h_sz, num_layers=2, batch_first=True, bidirectional=True)
        else:
            word_dim = self.word_dim
            # self.q_emb = nn.Identity()
            self.q_lstm = nn.GRU(word_dim, self.h_sz, num_layers=1, batch_first=True, bidirectional=True)
        self.q_MLP = nn.Linear(self.h_sz * 2, self.out_dim) #사용

        # self.o_encoder = nn.Sequential(
        #     nn.Embedding(len(self.vocab), self.out_dim, max_norm=1),
        #     nn.Linear(self.out_dim, self.out_dim),
        #     nn.ReLU(),
        # )
        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        if self.monolithic:
            self.qvo_fusion = nn.Sequential(nn.Linear(self.out_dim, self.max_val))
        else:
            #not implemented only monolithic case
            self.create_puzzle_tail(args)
        
        #vision_model frozen 

        for params in im_backbone.parameters():
            params.require_grads=False
        for params in self.word_embed_model.model.parameters():
            params.require_grads=False

    def create_puzzle_head(self):
        self.im_encoder = nn.Sequential( #2048, 128
            nn.Linear(self.im_feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
        )

    def create_puzzle_tail(self, args):
        self.puzzle_ids = args.puzzle_ids
        ans_decoder = [
            nn.Sequential(nn.Linear(self.out_dim, 1))
        ]  # start with a dummy as we are 1-indexed wrt puzzle ids.
        if args.puzzles == "all":
            puzzles = range(1, gv.num_puzzles + 1)
        else:
            puzzles = self.puzzle_ids
        for pid in puzzles:  # self.puzzle_ids:
            num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)] if args.loss_type == "classifier" else 1
            if int(pid) not in gv.SEQ_PUZZLES:
                ans_decoder.append(
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, num_classes),
                    )
                )
            else:
                ans_decoder.append(nn.LSTM(self.out_dim, num_classes, num_layers=1, batch_first=True))
        self.ans_decoder = nn.ModuleList(ans_decoder)

    def decode_image(self, im_list):
        """convert torch tensor images back to Image bcos VL FLAVA model works with images."""
        #        im_list = (im_list +1)/2. # this is in range [0, 1].
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))]  # convert im
        return im_list

    def save_grad_hook(self):
        self.vis_grad = None

        def bwd_hook(module, in_grad, out_grad):
            self.vis_grad = out_grad

        return bwd_hook

    def save_fwd_hook(self):
        self.vis_conv = None

        def fwd_hook(__, _, output):
            self.vis_conv = output

        return fwd_hook

    def encode_image(self, im, pids=None):
        if self.train_backbone:
            x = self.im_cnn(im).squeeze()
        else:
            with torch.no_grad():
                x = self.im_cnn(im).squeeze()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.use_single_image_head:
            y = self.im_encoder(x)
        else:
            y = torch.zeros(len(im), self.out_dim).cuda()
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.cuda()
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[int(self.puzzle_ids[t])](x[idx]))

        return y

    def decode_text(self, text):
        get_range = lambda x: range(1, x) if x < 70 else range(x - 70 + 4, x)
        tt = text.cpu()
        text = [
            " ".join([self.vocab.idx2word[int(j)] for j in tt[i][get_range(torch.nonzero(tt[i])[-1])]])
            for i in range(len(tt))
        ]
        return text

    def encode_text(self, text):
        if self.word_embed == "standard":
            x = self.q_emb(text)
            x, (h, _) = self.q_lstm(x.float())
            x = F.relu(self.q_MLP(x.mean(1)))
        elif self.word_embed == "gpt" or "bert" or "glove":
            text = self.decode_text(text)
            q_enc = torch.zeros(len(text), self.max_qlen, self.word_dim).cuda()
            for ii, tt in enumerate(text):
                q_feat = self.word_embed_model.word_embed(tt)
                q_enc[ii, : min(self.max_qlen, len(q_feat)), :] = q_feat
            x, (h, _) = self.q_lstm(q_enc.float())
            x = F.relu(self.q_MLP(x.mean(1)))
        else:
            x = self.word_embed_model.word_embed(text)

        return x

    def seq_decoder(self, decoder, feat):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(gv.MAX_DECODE_STEPS):
            try:
                out[k], hx = decoder(feat, hx)
            except:
                pdb.set_trace()
        return out

    def decode_individual_puzzles(self, feat, pids):
        upids = torch.unique(pids)
        out_feats = {}
        for t in range(len(upids)):
            idx = pids == upids[t]
            key = str(upids[t].item())
            key_idx = np.where(int(key) == np.array(self.sorted_puzzle_ids))[0][0] + 1  # +1 because we use 1-indexed.
            if upids[t] not in gv.SEQ_PUZZLES:
                out_feats[int(key)] = self.ans_decoder[key_idx](feat[idx])
            else:
                out_feats[int(key)] = self.seq_decoder(self.ans_decoder[key_idx], feat[idx])
        return out_feats

    def forward(self, return_loss=True, **sample):
        im = sample["images"]
        puzzle_ids = sample["pids"]
        q = sample["questions"]

        im_feat = self.encode_image(im, puzzle_ids)
        q_feat = self.encode_text(q)
        qv_feat = self.qv_fusion(torch.cat([im_feat, q_feat], dim=1))
        if self.monolithic:
            qv_feat = qv_feat.unsqueeze(1)
            qvo_feat = self.qvo_fusion(qv_feat).squeeze()
        else:
            qvo_feat = self.decode_individual_puzzles(qv_feat, puzzle_ids)

        loss = self.loss_fn(qvo_feat, sample["answers"].long()[:,0])

        output = {
            "logit" : qvo_feat,
            "loss" : loss
        }
        return output