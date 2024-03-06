import torch
import numpy as np
import sys
sys.path.append("/SeqMMLearning")
# from llava.model.lavis.blip2_origin import Blip2Base
from llava.model.multimodal_projector.builder import build_vision_projector
# from llava.model.lavis.models.blip2 import Blip2Base
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel
from transformers import AutoTokenizer
from mamba_ssm import Mamba


class SequentialMM_Model_legacy(nn.Module):
    def __init__(self, llm, query_num, args, device):
        super().__init__()
        self.llm = llm
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        self.llm_tokenizer.padding_side="right"
        self.llm_tokenizer.truncation_side="right"

        blip_model = Blip2Base()
        self.num_query_token = query_num
        print("init qformer")
        self.Qformer, self.query_tokens = blip_model.init_Qformer(
            self.num_query_token, 1024
        )

        self.Qformer.bert.embeddings.word_embeddings=None 
        self.Qformer.bert.embeddings.position_embeddings=None

        self.Qformer.cls=None

        # NOTE vicuna tokenizer init한것도 받아야함 (eos token)

        self.qformer_text_input = True #NOTE 실제 text은 안들어가지만 모델 구조단에서 필요할수도?

        print("init qformer finished")
        self.config = self.llm.config
        self.args = args
        self.bridge_former_to_projector = nn.Linear(768, 1024)
        print("build vision projector")
        mm_projector = build_vision_projector(self.llm.config)
        self.mm_projector = mm_projector
        self.device=device


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def load_mm_projector_state_dict(self):
        mm_projector_state_dict = torch.load(self.args.mm_projector_model_path)
        new_state_dict={}
        for key, value in mm_projector_state_dict.items():
            if key.startswith("model.mm_projector."):
                new_key = key[len("model.mm.projector."):]
                new_state_dict[new_key] = value
        self.mm_projector.load_state_dict(new_state_dict)


    def forward(self, return_loss=True, **sample):
        image_embeds = sample["image_feat"] # list[ [num_img, 576], [num_img, 576], [num_img, 576] ...] -> len : B / list이므로 gpu로 못올림 
        img_nums = sample["image_num_in_batch"]

        batch_size = len(image_embeds)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1) #[B, query_num, qformer_dim] -> 여기서 하나씩 slice해서 for문에서 사용하면 됨
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device) #[B, query_num]

        b_query_output = []
        for batch_idx in range(batch_size):
            # print(f"batch inside {batch_idx}, device {self.device}")
            # print(f"batch inside {batch_idx}, img num : {img_nums[batch_idx]}, device {self.device}")
            feat = image_embeds[batch_idx].to(self.device) #[num_img, 576, 1024] tensor 
            feat_atts = torch.ones(feat.size()[:-1], dtype=torch.long).to(self.device) #[num_img, 576] tensor
            img_num = img_nums[batch_idx]

            for img_idx in range(img_num):
                # print(f"img idx : {img_idx} in batch {batch_idx} of {self.device}")
                #만약 처음이면 instruction image 없이 qformer 통과하기 
                if img_idx == 0:
                    # print(f"qformer in {self.device}")
                    query_output = self.Qformer.bert(
                        input_ids=None, #NOTE 없어도 돌아가는지 확인
                        attention_mask=query_atts[batch_idx].unsqueeze(0), #[1, query_num] #원래 코드는 batch단위이기 때문에 unsqueeze해줘야함 
                        query_embeds = query_tokens[batch_idx].unsqueeze(0), #[1, query_num, qformer_dim]
                        encoder_hidden_states = feat[img_idx].unsqueeze(0), #[1, 576, 1024]
                        encoder_attention_mask=feat_atts[img_idx].unsqueeze(0) #[1, 576]
                    ).last_hidden_state[:, :query_tokens.size(1),:] #[1, query_num, qformer_dim]
                    instruction_feat = query_output #cache
                    # print(f"qformer_out {self.device}")

                else: #처음이 아닌경우 -> instruction_feat이 instruction 자리에 들어간다!
                    #NOTE query attention값을 새로 만들어야함
                    # print(f"second_qformer_in {self.device}")
                    instruction_att_mask = torch.ones(instruction_feat.size()[:-1], dtype=torch.long).to(self.device) #[1, query_num]
                    Qformer_atts = torch.cat([query_atts[batch_idx].unsqueeze(0), instruction_att_mask], dim=1) #[1, query_num + instruction_feat] -> 걍 query_num*2임 
                    # print(f"{self.device} instruction_feat.shape : {instruction_feat.shape}")
                    # print(f"{self.device} Qformer_atts.shape : {Qformer_atts.shape}")
                    # print(f"{self.device} feat[img_idx].shape : {feat[img_idx].shape}")
                    query_output = self.Qformer.bert(
                        input_ids = instruction_feat,
                        attention_mask=Qformer_atts,
                        query_embeds = query_tokens[batch_idx].unsqueeze(0), #[1, query_num, qformer_dim]
                        encoder_hidden_states = feat[img_idx].unsqueeze(0), #[1, 576, 1024]
                        encoder_attention_mask=feat_atts[img_idx].unsqueeze(0) #[1, 576]
                    ).last_hidden_state[:, :query_tokens.size(1),:] #[1, query_num, qformer_dim]
                    instruction_feat = query_output
                    # print(f"second_qformer_out {self.device}")
            
            #다 돌고난 후 query_output을 batch단위로 다시 만들기 위해 stack
            b_query_output.append(query_output.squeeze(0))
        #     print(f"b_query_output {len(b_query_output)} , device {self.device}")
        # print(f"end {self.device}")
        b_query_output = torch.stack(b_query_output, dim=0) #[B, query_num, qformer_dim] #NOTE device check
        qformer_output = self.mm_projector(self.bridge_former_to_projector(b_query_output)) #[B, query_num, 4096]
        qformer_llm_atts = torch.ones(qformer_output.size()[:-1], dtype=torch.long).to(self.device)

        #이제 text input 만들기
        text_input_tokens = self.llm_tokenizer(
            sample["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in sample['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill( 
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(qformer_llm_atts.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm.get_input_embeddings()(llm_tokens['input_ids']) #NOTE 이게 어떻게 batch단위로 되는지 확인하기 
        inputs_embeds = torch.cat([qformer_output, inputs_embeds], dim=1)
        attention_mask = torch.cat([qformer_llm_atts, llm_tokens['attention_mask']], dim=1)

        outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        return outputs



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



class SequentialMM_Model(nn.Module):
    def __init__(self, llm, query_num, args, device):
        super().__init__()
        self.llm = llm
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        self.llm_tokenizer.padding_side="right"
        self.llm_tokenizer.truncation_side="right"
        self.mamba_patch = Mamba(
            d_model=768,
            d_state=4,
            d_conv=4,
            expand=2,
        )

        self.mamba_img_seq = Mamba(
            d_model=768,
            d_state=4,
            d_conv=2,
            expand=2,
        )
        self.ln_vision = nn.Linear(1024, 768)
        print("init mamba finished")
        self.config = self.llm.config
        self.args = args
        self.bridge_former_to_projector = nn.Linear(768, 1024)
        print("build vision projector")
        mm_projector = build_vision_projector(self.llm.config)
        self.mm_projector = mm_projector
        self.device=device


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def load_mm_projector_state_dict(self):
        mm_projector_state_dict = torch.load(self.args.mm_projector_model_path)
        new_state_dict={}
        for key, value in mm_projector_state_dict.items():
            if key.startswith("model.mm_projector."):
                new_key = key[len("model.mm.projector."):]
                new_state_dict[new_key] = value
        self.mm_projector.load_state_dict(new_state_dict)


    def forward(self, return_loss=True, **sample):
        image_embeds = sample["image_feat"].to(self.device) # list[ [num_img, 576], [num_img, 576], [num_img, 576] ...] -> len : B / list이므로 gpu로 못올림 
        img_nums = sample["image_num_in_batch"]
        image_embeds = self.ln_vision(image_embeds) # [B, max_img, token, 768]

        B, max_img, token_num, D = image_embeds.shape 
        image_embeds = image_embeds.reshape(B*max_img, token_num, D)



        mamba_patch_embed = self.mamba_patch(image_embeds)[:,-1, :] #[B*max_img, D]

        mamba_patch_embed = mamba_patch_embed.reshape(B, max_img, D)

        #mamba forward
        mamba_embed = self.mamba_img_seq(mamba_patch_embed) #[B, max_img, D]

        #마지막 값들만 배치 단위로 가져와야함
        batch_indices = torch.arange(B)
        adjusted_lengths = torch.tensor(img_nums) - 1
        mamba_final_states = mamba_embed[batch_indices, adjusted_lengths] #[B, D]
        mamba_final_states = self.mm_projector(self.bridge_former_to_projector(mamba_final_states)) #[B,4096]

        mamba_final_states = mamba_final_states.unsqueeze(1) #[B, 1, 4096]
        mamba_atts = torch.ones(mamba_final_states.size()[:-1], dtype=torch.long).to(self.device)

        #이제 text input 만들기
        text_input_tokens = self.llm_tokenizer(
            sample["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in sample['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill( 
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(mamba_atts.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm.get_input_embeddings()(llm_tokens['input_ids']) #NOTE 이게 어떻게 batch단위로 되는지 확인하기 
        inputs_embeds = torch.cat([mamba_final_states, inputs_embeds], dim=1)
        attention_mask = torch.cat([mamba_atts, llm_tokens['attention_mask']], dim=1)

        outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        return outputs



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


class only_LLM(nn.Module):
    def __init__(self, llm, query_num, args, device):
        super().__init__()
        self.llm = llm
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        self.llm_tokenizer.padding_side="right"
        self.llm_tokenizer.truncation_side="right"
        self.device=device

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


    def forward(self, return_loss=True, **sample):


        #이제 text input 만들기
        text_input_tokens = self.llm_tokenizer(
            sample["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in sample['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill( 
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm.get_input_embeddings()(llm_tokens['input_ids']) 
        outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=llm_tokens['attention_mask'],
                return_dict=True,
                labels=targets,
            )
        
        return outputs



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
