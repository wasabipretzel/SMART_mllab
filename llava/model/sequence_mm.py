import torch
import numpy as np
import sys
sys.path.append("/SeqMMLearning")
# from llava.model.lavis.blip2_origin import Blip2Base
from llava.model.multimodal_projector.builder import build_vision_projector
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel
from transformers import PreTrainedModel
from transformers import AutoTokenizer



class SequentialMM_Model(nn.Module):
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
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_query_token, configuration_qformer.hidden_size)
            )
        else:
            configuration_qformer = InstructBlipQFormerConfig.from_pretrained(args.pretrained_qformer_path)
            self.qformer = InstructBlipQFormerModel.from_pretrained(args.pretrained_qformer_path) #encoder hidden size 1408
            self.ln_vision = nn.Linear(1024, 1408)
            self.query_tokens = nn.Parameter(torch.load(args.pretrained_qformer_query_token_path)) #[1, 32, 768]

    
        self.query_tokens.data.normal_(mean=0.0, std=configuration_qformer.initializer_range)
        print("init qformer finished")
        self.config = self.llm.config
        # self.peft_config = self.llm.peft_config
        # self.tokenizer = init_tokenizer()
        self.args = args
        self.bridge_former_to_projector = nn.Linear(768, 1024)
        print("build vision projector")
        mm_projector = build_vision_projector(self.llm.config)
        self.mm_projector = mm_projector
        self.device = device
    
    def load_mm_projector_state_dict(self):
        mm_projector_state_dict = torch.load(self.args.mm_projector_model_path)
        new_state_dict={}
        for key, value in mm_projector_state_dict.items():
            if key.startswith("model.mm_projector."):
                new_key = key[len("model.mm.projector."):]
                new_state_dict[new_key] = value
        self.mm_projector.load_state_dict(new_state_dict)
    
    def process_input_llm(self,query_output, text_token_embeds, target, img_nums, input_ids_len):
        """
            query_output : [B, maximg*query, 4096]
            text_token_embeds : [B, maxS, 4096]
            target : [[S1], [S2] ..] list (inside tensor)
        """
        # 각 이미지당 token개수
        img_nums = [img*self.num_query_token for img in img_nums]
        total_token_len = [] # 각 batch별로 자르기 위함
        for num_img, len_text in zip(img_nums, input_ids_len):
            total_token_len.append(num_img+len_text)
        max_token_len = max(total_token_len)
        llm_input_rep = []
        llm_target = []
        B = query_output.shape[0]
        dim = query_output.shape[-1]
        for each_batch in range(B):
            #query value
            #text value
            each_batch_query = query_output[each_batch][:img_nums[each_batch],:] #[each_img*query, 4096]
            each_batch_input_ids = text_token_embeds[each_batch][:input_ids_len[each_batch],:] #[each_len, 4096]

            each_batch_llm_input=torch.cat([each_batch_query, each_batch_input_ids], dim=0) #[each_img*query+each_len, 4096]
            query_target = torch.tensor([-100]*each_batch_query.shape[0]).to(self.device)
            each_batch_llm_target=torch.cat([query_target, target[each_batch]]) #[S1]

            #add padding mask
            cur_len = each_batch_llm_input.shape[0]
            #[max_token_len - cur_len, 4096]
            pad_rep_mask =  torch.zeros([max_token_len-cur_len, dim], dtype=torch.long).to(self.device)
            each_batch_llm_input = torch.cat([each_batch_llm_input, pad_rep_mask], dim=0) #[each_img*query+each_len+padmask, 4096]
            llm_input_rep.append(each_batch_llm_input)

            #target padding mask
            add_len = max_token_len - each_batch_llm_target.shape[0]
            add_pad_mask = torch.full([add_len], -100).to(self.device)
            each_batch_llm_target = torch.cat([each_batch_llm_target, add_pad_mask])
            llm_target.append(each_batch_llm_target)

        llm_input_rep = torch.stack(llm_input_rep, dim=0) #[B, total_token_len, 4096]

        llm_input_pad_mask = llm_input_rep.ne(0).any(dim=2)

        llm_target = torch.stack(llm_target, dim=0)

        #target은 기존 앞에 
        assert llm_input_rep.shape[1] == llm_input_pad_mask.shape[1] == llm_target.shape[1]

        return llm_input_rep, llm_input_pad_mask, llm_target

    def forward(self, return_loss=True, **sample): #return_loss=True로 해야 huggingface trainer eval때 loss을 return해줌

        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]
        input_ids_len = sample["input_ids_len"]

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
        # if not self.args.use_pretrained_qformer:
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

        qformer_output = self.mm_projector(self.bridge_former_to_projector(query_output)) #[B, max_img*self.num_query_token, 4096]




        #input_text_ids embedding
        text_input_ids = sample["input_ids"] #[B, maxS] -> [B*maxS] -> [B, maxS, 4096]
        B, maxS = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(B*maxS)
        text_token_embeds = self.llm.get_input_embeddings()(text_input_ids) #[B*maxS, 4096]
        text_token_embeds = text_token_embeds.reshape(B, maxS, -1) #[B, maxS, 4096] 

        target = sample["target_ids"]
        llm_input_rep, llm_input_pad_mask, llm_target = self.process_input_llm(qformer_output, text_token_embeds, target, img_nums, input_ids_len)

        result = self.llm(
            attention_mask=llm_input_pad_mask,
            inputs_embeds = llm_input_rep,
            labels=llm_target,
            return_dict=True
        )


        return result #[B*max_img, query_token, 768]

    @torch.no_grad()
    def predict(self, **sample):

        image_embeds = sample["images"].to(self.device) #[B, max_img, num_token, D]
        images_atts = sample["images_att"].to(self.device) #[B*max_img, num_token]
        img_nums = sample["image_num_in_batch"]
        input_ids_len = sample["input_ids_len"]

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
        # if not self.args.use_pretrained_qformer:
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

        qformer_output = self.mm_projector(self.bridge_former_to_projector(query_output)) #[B, max_img*self.num_query_token, 4096]




        #input_text_ids embedding
        text_input_ids = sample["input_ids"] #[B, maxS] -> [B*maxS] -> [B, maxS, 4096]
        B, maxS = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(B*maxS)
        text_token_embeds = self.llm.get_input_embeddings()(text_input_ids) #[B*maxS, 4096]
        text_token_embeds = text_token_embeds.reshape(B, maxS, -1) #[B, maxS, 4096] 

        target = sample["target_ids"]
        llm_input_rep, llm_input_pad_mask, llm_target = self.process_input_llm(qformer_output, text_token_embeds, target, img_nums, input_ids_len)


        result = self.llm.generate( #110M
            attention_mask=llm_input_pad_mask,
            inputs_embeds=llm_input_rep,
            return_dict=True
        )

        return result