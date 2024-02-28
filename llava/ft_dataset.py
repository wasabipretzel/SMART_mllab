"""
    {
        "images" : [B, MAXIMAGE, TOKENS, D],
        "images_att" : [B*MAXIMAGE, TOKENS],
        "image_num_in_batch" : [20, 12, 8, ..] -> len : batch num
    }
    을 return하는 dataset class
    +
    이를 qformer에 output query feature까지 만드는 코드
"""
import os

import numpy as np
import torch 
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader
from model.lavis.blip2_origin import Blip2Base
from torch import nn




class SeqMM_FT_Dataset(Dataset):
    def __init__(self, candidates, feature_path, is_train=True):
        self.serial_candidates = candidates
        self.is_train = is_train 
        self.feature_path = feature_path
    

    def __len__(self):
        return len(self.serial_candidates)

    
    def __getitem__(self, index):
        # 하나의 manual에 대한 이미지 feature들을 불러와서 image 개수와 함께 return
        single_feature_path = os.path.join(self.feature_path, self.serial_candidates[index]+'.npy')
        feat = torch.tensor(np.load(single_feature_path)) #[num_img, 576]
        num_img = feat.shape[0]
        result = {
            "feat" : feat, #tensor
            "num_img" : num_img, #int
            
        }
        return result
    



def collate_fn(batch):
    # batch을 돌면서 [B, MaxImage, token, D] 을 만들고 dim=1 부분 mask
    # qformer encoder attention 으로 쓸 [B*MaxImage, token] mask 
    b_feat = []
    b_img_nums = []
    b_feat_mask = [] #[B*max_img, num_token] ->  feat넘길때 넘긴다음 전체 stack해버리면 될거같은데 
    for each_batch in batch:
        b_img_nums.append(each_batch["num_img"])
    b_max_img = max(b_img_nums)
    _, num_token, D = batch[0]["feat"].shape


    for each_batch in batch:
        if each_batch["feat"].shape[0] == b_max_img:
            b_feat.append(each_batch["feat"])
            att_mask = torch.ones((b_max_img, num_token))
            b_feat_mask.append(att_mask)
        else:
            #부족한 만큼 padding 
            each_feat = each_batch["feat"] #[num_img, num_token, D]
            att_mask = torch.ones((each_feat.shape[0], num_token)) #[num_img, num_token]
            marginal = torch.zeros((b_max_img - each_feat.shape[0], num_token, D))
            att_mask_zero = torch.zeros((b_max_img - each_feat.shape[0], num_token))
            each_feat = torch.cat([each_feat, marginal], dim=0)
            att_mask = torch.cat([att_mask, att_mask_zero], dim=0)
            b_feat.append(each_feat)
            b_feat_mask.append(att_mask)
    
    b_feat = torch.stack(b_feat) #[B, max_img, num_token, D]
    b_feat_mask = torch.stack(b_feat_mask, dim=0) #[B, max_img, num_token]이 되도록
    
    result = {
        "images" : b_feat, #[B, max_img, num_token, D]
        "images_att" : b_feat_mask, #[B, max_img, num_token]
        "image_num_in_batch" : b_img_nums
    }
    return result



### model part
class Model(Blip2Base):
    def __init__(self, query_num,visual_width):
        super().__init__()
        self.num_query_token = query_num
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, visual_width
        )
        # self.ln_vision = nn.Linear(visual_width, 768)

    def forward(self, sample):
        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]

        #text input
        # text_input = sample["text_input"] #input ids [B, max_Seq]
        # text_atts = sample["text_att"] #[B, maxSeq]
        text_input = torch.randint(0, 10, (4, 80)).to(self.device)
        text_atts = torch.ones(4, 80).to(self.device)

        #image embedding
        # image_embeds = self.ln_vision(images_feat) 
        B, max_img, num_token, D = image_embeds.shape #D :768
        _, max_seq = text_input.shape

        #image / text input, attention mask expand
        image_embeds = image_embeds.reshape(B*max_img, num_token, D) #[B*max_img, num_token, D]
        images_atts = images_atts.reshape(B*max_img, -1) #[B*max_img, num_token]
        text_input_expand = text_input.repeat_interleave(max_img,dim=0) #[B*max_img, maxSeq] #NOTE 이거 [1,2,3]이면 [1,1,1, 2,2,2, 3,3,3] 이런식으로 반복해야하는데 맞게 들어가는지 확인해야함
        text_atts_expand = text_atts.repeat_interleave(max_img,dim=0) #NOTE 이것도 확인해야함

        #query token / mask expand, 
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [B*max_img, querynum, D]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device) #[B*max_img, query_num] #NOTE 여기도 마스킹 해줘야함!!!
        #NOTE images_atts에서 maxSeq을 query_num만큼 잘라서 가져오면 될거같음
        query_atts = images_atts[:,:self.num_query_token]

        #get final attention mask
        #qformer attention mask
        qformer_atts = torch.cat([query_atts, text_atts_expand], dim=1) #[B*max_img, query_num + maxSeq]


        query_output = self.Qformer.bert(
            text_input_expand, #[B*max_img, maxSeq]
            attention_mask=qformer_atts, #[B*max_img, query_num + maxSeq]
            query_embeds=query_tokens, #[B*max_img, querynum, D]
            encoder_hidden_states=image_embeds, #[B*max_img, num_token, 1024]
            encoder_attention_mask= images_atts, #[B*max_img, num_token]
            return_dict=True,
        ).last_hidden_state[:,:query_tokens.size(1), :]

        #query_output.last_hidden_state[:,:query_tokens.size(1),:] #NOTE 확인한번하기 

        #NOTE [B, max_img*query_token, 768] , mask도 동일하게 처리 
        output_flatten = query_output.reshape(B, max_img*self.num_query_token, -1) #[B, max_img*self.num_query_token, D]
        output_mask = query_atts.reshape(B, max_img*self.num_query_token) #[B, max_img*self.num_query_token]

        return output_flatten, output_mask #[B*max_img, query_token, 768]

        





        







if __name__ == '__main__':
    
    # NOTE : cuda 올리는 작업 해야함
    base_path = "/data/dataset/manuals"
    feat_path = "/data/dataset/features"
    device = 'cuda:3'
    query_num = 16
    visual_width = 1024

    serial_nums = os.listdir(base_path)
    dataset = SeqMM_FT_Dataset(serial_nums, feat_path)

    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x),
    )

    model = Model(query_num, visual_width)
    model.to(device)
    for batch_idx, sample in enumerate(train_dataloader):

        output = model(sample)

