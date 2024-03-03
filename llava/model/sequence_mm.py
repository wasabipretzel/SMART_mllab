import torch
import numpy as np
from llava.model.lavis.blip2_origin import Blip2Base
from llava.model.multimodal_projector.builder import build_vision_projector
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel
from transformers import PreTrainedModel


class SequentialMM_Model(Blip2Base):
    def __init__(self, llm, query_num, args, device):
        super().__init__()
        self.llm = llm
        self.num_query_token = query_num
        print("init qformer")
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     self.num_query_token, 1024, self.device
        # )
        if not args.use_pretrained_qformer:
            configuration_qformer = InstructBlipQFormerConfig.from_pretrained('Salesforce/instructblip-vicuna-7b')
            configuration_qformer.num_hidden_layers = 4
            configuration_qformer.num_hidden_layers = 8
            configuration_qformer.encoder_hidden_size=768
            self.qformer = InstructBlipQFormerModel(configuration_qformer)
            self.ln_vision = nn.Linear(1024, 768)
        else:
            configuration_qformer = InstructBlipQFormerConfig.from_pretrained(args.pretrained_qformer_path)
            self.qformer = InstructBlipQFormerModel.from_pretrained(args.pretrained_qformer_path) #encoder hidden size 1408

        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_token, configuration_qformer.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=configuration_qformer.initializer_range)
        print("init qformer finished")
        self.config = self.llm.config
        # self.peft_config = self.llm.peft_config
        # self.tokenizer = init_tokenizer()
        self.args = args
        self.qformer_to_mm_projector = nn.Linear(768, 1024)
        print("build vision projector")
        mm_projector = build_vision_projector(self.llm.config)
        self.mm_projector = mm_projector
    
    def load_mm_projector_state_dict(self):
        mm_projector_state_dict = torch.load(self.args.mm_projector_model_path)
        new_state_dict={}
        for key, value in mm_projector_state_dict.items():
            if key.startswith("model.mm_projector."):
                new_key = key[len("model.mm.projector."):]
                new_state_dict[new_key] = value
        self.mm_projector.load_state_dict(new_state_dict)

    def forward(self, return_loss=True, **sample): #return_loss=True로 해야 huggingface trainer eval때 loss을 return해줌

        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]

        #text input
        text_input = sample["qformer_ids"] #input ids [B, max_Seq]
        text_atts = torch.ne(text_input, 0)
        # text_atts = sample["text_att"] #[B, maxSeq]
        #image embedding
        # image_embeds = self.ln_vision(images_feat) 
        B, max_img, num_token, D = image_embeds.shape #D :768
        _, max_seq = text_input.shape

        #image / text input, attention mask expand
        image_embeds = image_embeds.reshape(B*max_img, num_token, D) #[B*max_img, num_token, D]
        if args.use_pretrained_qformer:
            image_embeds = self.ln_vision(image_embeds)
        images_atts = images_atts.reshape(B*max_img, -1) #[B*max_img, num_token]
        text_input = text_input.repeat_interleave(max_img,dim=0) #[B*max_img, maxSeq] #NOTE 이거 [1,2,3]이면 [1,1,1, 2,2,2, 3,3,3] 이런식으로 반복해야하는데 맞게 들어가는지 확인해야함
        text_atts = text_atts.repeat_interleave(max_img,dim=0) #NOTE 이것도 확인해야함

        #query token / mask expand, 
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [B*max_img, querynum, D]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device) #[B*max_img, query_num] #NOTE 여기도 마스킹 해줘야함!!!
        #NOTE images_atts에서 maxSeq을 query_num만큼 잘라서 가져오면 될거같음
        query_atts = images_atts[:,:self.num_query_token]

        #get final attention mask
        #qformer attention mask
        qformer_atts = torch.cat([query_atts, text_atts], dim=1) #[B*max_img, query_num + maxSeq]


        query_output = self.qformer(
            text_input, #[B*max_img, maxSeq]
            attention_mask=qformer_atts, #[B*max_img, query_num + maxSeq]
            query_embeds=query_tokens, #[B*max_img, querynum, D]
            encoder_hidden_states=image_embeds, #[B*max_img, num_token, 1024]
            encoder_attention_mask= images_atts, #[B*max_img, num_token]
            return_dict=True,
        ).last_hidden_state[:,:query_tokens.size(1), :]

        #query_output.last_hidden_state[:,:query_tokens.size(1),:] #NOTE 확인한번하기 

        #NOTE [B, max_img*query_token, 768] , mask도 동일하게 처리 
        query_output = query_output.reshape(B, max_img*self.num_query_token, -1) #[B, max_img*self.num_query_token, D]
        query_atts = query_atts.reshape(B, max_img*self.num_query_token) #[B, max_img*self.num_query_token]

        qformer_output = self.mm_projector(self.qformer_to_mm_projector(query_output)) #[B, max_img*self.num_query_token, 4096]

        #input_text_ids embedding
        text_input_ids = sample["input_ids"] #[B, maxS] -> [B*maxS] -> [B, maxS, 4096]
        B, maxS = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(B*maxS)
        text_token_embeds = self.llm.get_input_embeddings()(text_input_ids) #[B*maxS, 4096]
        text_token_embeds = text_token_embeds.reshape(B, maxS, -1) #[B, maxS, 4096] 

        #special_token embedding
        img_st = sample["im_start_ids"]
        img_end = sample["im_end_ids"]

        img_st_emb = self.llm.get_input_embeddings()(img_st) #[spl_token_len, 4096]
        img_end_emb = self.llm.get_input_embeddings()(img_end) #[spl_token_len, 4096]

        img_st_emb = img_st_emb.unsqueeze(0).expand(B, -1, -1) 
        img_end_emb = img_end_emb.unsqueeze(0).expand(B, -1, -1)

        img_st_att = torch.ones(img_st_emb.size()[:-1], dtype=torch.long).to(img_st_emb.device)
        img_end_att = torch.ones(img_end_emb.size()[:-1], dtype=torch.long).to(img_st_emb.device)

        #=====================vicuna input===============================
        input_embeds = torch.cat([img_st_emb, qformer_output, img_end_emb, text_token_embeds], dim=1)
        input_atts = torch.cat([img_st_att, query_atts, img_end_att, sample["input_ids_pad_mask"]], dim=1) 
        #NOTE dimension check

        #target
        # img_st_emb, qformer img_end_emb sequence length 만큼 -100을 추가로 채우면 되나?
        target = sample["target_ids"] #padding mask -100으로 해야함 
        extra_target= torch.tensor([-100] * (img_st_emb.shape[1] + qformer_output.shape[1] + img_end_emb.shape[1])).unsqueeze(0).expand(B, -1).to(self.device) #[B, preceding length]
        # target = extra_target + target 
        target = torch.cat([extra_target, target], dim=1)
        #NOTE length check : target이랑 input_embeds랑 
        result = self.llm( #110M
            attention_mask=input_atts,
            inputs_embeds=input_embeds,
            labels=target,
            return_dict=True
        )

        #img_st_emb을 batch 만큼 확장 + qformer_output + img_end_emb을 batch만큼 확장 + text_token_embeds
        #확장한거 ne + qformer_atts + 확장한거 ne + text padding
        #target도 만들어야함(얘는 -100)

        # text_input_embeds = self.llm.embed_tokens(text_input_ids)
        #NOTE 
        # 1. bin file load -> ok
        # 2. position_ids?
        # 3. mm_project 통과
        # 4. cat 후에 self.llm -> input_ids은 None이고, input_embeds에 다 넣고 label을 target으로 넣으면 됨
        #NOTE -> mm_projector통과시켜야함 -> load 
        #NOTE -> text embedding 값 받아야함 -> self.llm.embed_tokens(input_ids)하면됨 , img_st, img_end 도 embed_tokens해야되네
        #img_st : [1] -> [B,  [img_st, query, img_end, ]
        # result_tensor = torch.cat((specific_tensor.unsqueeze(0).expand(original_tensor.size(0), -1, -1), original_tensor, specific_tensor.unsqueeze(0).expand(original_tensor.size(0), -1, -1)), dim=1)


        return result #[B*max_img, query_token, 768]

    @torch.no_grad()
    def predict(self, **sample):
        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]

        #text input
        text_input = sample["qformer_ids"] #input ids [B, max_Seq]
        text_atts = torch.ne(text_input, 0)
        # text_atts = sample["text_att"] #[B, maxSeq]
        #image embedding
        # image_embeds = self.ln_vision(images_feat) 
        B, max_img, num_token, D = image_embeds.shape #D :768
        _, max_seq = text_input.shape

        #image / text input, attention mask expand
        image_embeds = image_embeds.reshape(B*max_img, num_token, D) #[B*max_img, num_token, D]
        image_embeds = self.ln_vision(image_embeds)
        images_atts = images_atts.reshape(B*max_img, -1) #[B*max_img, num_token]
        text_input = text_input.repeat_interleave(max_img,dim=0) #[B*max_img, maxSeq] #NOTE 이거 [1,2,3]이면 [1,1,1, 2,2,2, 3,3,3] 이런식으로 반복해야하는데 맞게 들어가는지 확인해야함
        text_atts = text_atts.repeat_interleave(max_img,dim=0) #NOTE 이것도 확인해야함

        #query token / mask expand, 
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [B*max_img, querynum, D]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device) #[B*max_img, query_num] #NOTE 여기도 마스킹 해줘야함!!!
        #NOTE images_atts에서 maxSeq을 query_num만큼 잘라서 가져오면 될거같음
        query_atts = images_atts[:,:self.num_query_token]

        #get final attention mask
        #qformer attention mask
        qformer_atts = torch.cat([query_atts, text_atts], dim=1) #[B*max_img, query_num + maxSeq]


        query_output = self.qformer(
            text_input, #[B*max_img, maxSeq]
            attention_mask=qformer_atts, #[B*max_img, query_num + maxSeq]
            query_embeds=query_tokens, #[B*max_img, querynum, D]
            encoder_hidden_states=image_embeds, #[B*max_img, num_token, 1024]
            encoder_attention_mask= images_atts, #[B*max_img, num_token]
            return_dict=True,
        ).last_hidden_state[:,:query_tokens.size(1), :]

        #query_output.last_hidden_state[:,:query_tokens.size(1),:] #NOTE 확인한번하기 

        #NOTE [B, max_img*query_token, 768] , mask도 동일하게 처리 
        query_output = query_output.reshape(B, max_img*self.num_query_token, -1) #[B, max_img*self.num_query_token, D]
        query_atts = query_atts.reshape(B, max_img*self.num_query_token) #[B, max_img*self.num_query_token]

        qformer_output = self.mm_projector(self.qformer_to_mm_projector(query_output)) #[B, max_img*self.num_query_token, 4096]

        #input_text_ids embedding
        text_input_ids = sample["input_ids"] #[B, maxS] -> [B*maxS] -> [B, maxS, 4096]
        B, maxS = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(B*maxS)
        text_token_embeds = self.llm.get_input_embeddings()(text_input_ids) #[B*maxS, 4096]
        text_token_embeds = text_token_embeds.reshape(B, maxS, -1) #[B, maxS, 4096] 

        #special_token embedding
        img_st = sample["im_start_ids"]
        img_end = sample["im_end_ids"]

        img_st_emb = self.llm.get_input_embeddings()(img_st) #[spl_token_len, 4096]
        img_end_emb = self.llm.get_input_embeddings()(img_end) #[spl_token_len, 4096]

        img_st_emb = img_st_emb.unsqueeze(0).expand(B, -1, -1) 
        img_end_emb = img_end_emb.unsqueeze(0).expand(B, -1, -1)

        img_st_att = torch.ones(img_st_emb.size()[:-1], dtype=torch.long).to(img_st_emb.device)
        img_end_att = torch.ones(img_end_emb.size()[:-1], dtype=torch.long).to(img_st_emb.device)

        #=====================vicuna input===============================
        input_embeds = torch.cat([img_st_emb, qformer_output, img_end_emb, text_token_embeds], dim=1)
        input_atts = torch.cat([img_st_att, query_atts, img_end_att, sample["input_ids_pad_mask"]], dim=1) 
        #NOTE dimension check

        #target
        # img_st_emb, qformer img_end_emb sequence length 만큼 -100을 추가로 채우면 되나?
        target = sample["target_ids"] #padding mask -100으로 해야함 
        extra_target= torch.tensor([-100] * (img_st_emb.shape[1] + qformer_output.shape[1] + img_end_emb.shape[1])).unsqueeze(0).expand(B, -1).to(self.device) #[B, preceding length]
        # target = extra_target + target 
        target = torch.cat([extra_target, target], dim=1)

        result = self.llm.generate( #110M
            attention_mask=input_atts,
            inputs_embeds=input_embeds,
            return_dict=True
        )