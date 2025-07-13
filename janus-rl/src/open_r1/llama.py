import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor
from models.modeling_vlm import MultiModalityPreTrainedModel
import PIL
from PIL import Image
from tqdm import tqdm
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List

from tqdm import tqdm

from torch import nn
from torchvision import transforms

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126  
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class JanusLLamaModel(MultiModalityCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config

    def save_stack_images(self, images: np.ndarray, batch_size: int, save_path: str, height: int = 384, weight: int = 384):
        blank_image = np.zeros((height, weight, 3), dtype=np.uint8)

        image_num_per_row = int(np.sqrt(batch_size).item())
        image_num_per_column = int(np.ceil(batch_size / image_num_per_row).item())

        images_to_padding = image_num_per_row * image_num_per_column - batch_size
        
        if images_to_padding != 0:
            images = np.concatenate([images, [blank_image] * images_to_padding], axis=0)

        rows = []
        for idx in range(0, image_num_per_row * image_num_per_column, image_num_per_row):
            row = np.hstack(images[idx:idx+image_num_per_row])
            rows.append(row)
        combined_image = np.vstack(rows)

        pil_image = Image.fromarray(combined_image)
        pil_image.save(save_path)
    
    @torch.no_grad()
    def generate_with_refine(
        self,
        vl_chat_processor: VLChatProcessor,
        input_ids,
        attention_mask,
        temperature: float = 1,
        parallel_size: int = 4,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        img_top_k: int = None,
        img_top_p: float = None,
        txt_top_k: int = None,
        txt_top_p: float = None,
        max_reflect_len: int = 80,
        task_list: List[int] = [1,2,3],
        cur_step=0,
        img_path: str=None,
    ):
        prompt = [
            '<end_of_image>\nLet me think Does this image match the prompt...',
            '<｜end▁of▁sentence｜>\nNext, I will draw a new image<begin_of_image>'
        ]
        all_imgs_1,img_ids_1,embeds_1,attention_mask_1 = [],[],[],[]
        output_text_ids,selfcheck,attention_mask_txt,embeds_2,attention_mask_2 = [],[],[],[],[]
        embeds_3,attention_mask_3,img_ids_2,all_imgs_2 = [],[],[],[]
        parallel_size = input_ids.shape[0]
        if 1 <= task_list[-1]:
            tokens = torch.repeat_interleave(input_ids,2,dim=0)
            for i in range(tokens.size(0)):
                if i % 2 != 0:
                    pad_list = torch.where(tokens[i]==vl_chat_processor.pad_id)[0]
                    if pad_list.shape[0]==0:
                        st = 1
                    else:
                        st = pad_list[-1].item()+2
                    tokens[i, st:-1] = vl_chat_processor.pad_id
            inputs_embeds = self.language_model.get_input_embeddings()(tokens) 
            embeds_1 = inputs_embeds
            attention_mask_1 = torch.repeat_interleave(attention_mask, 2, dim=0) 
            cur_atten_mask = attention_mask_1
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
            for i in tqdm(range(image_token_num_per_image)):
                outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=cur_atten_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state
                logits = self.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                if img_top_k:
                    v, _ = torch.topk(logits, min(img_top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits / temperature, dim=-1)
                if img_top_p:
                    probs_sort, probs_idx = torch.sort(probs,
                                                    dim=-1,
                                                    descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    mask = probs_sum - probs_sort > img_top_p
                    probs_sort[mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
                cur_atten_mask = torch.cat([cur_atten_mask, torch.ones(cur_atten_mask.size(0), 1).to(attention_mask)], dim=1)
            dec = self.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            for i in range(parallel_size):
                all_imgs_1.append(PIL.Image.fromarray(visual_img[i]))
            img_ids_1 = generated_tokens

        if 2 <= task_list[-1]:
            inputs_embeds = embeds_1[::2,:,:] 
            under_embeds = torch.zeros((parallel_size, image_token_num_per_image, 4096), dtype=torch.bfloat16).cuda()
            for i in range(parallel_size):
                img_prompt = "<image_placeholder>"
                prepare_inputs = vl_chat_processor(
                    prompt=img_prompt, images=[all_imgs_1[i]], force_batchify=True
                ).to(input_ids.device)
                img_embeds = self.prepare_inputs_embeds(**prepare_inputs) 
                img_embeds = img_embeds[:,2:-1,:] 
                under_embeds[i,:,:] = img_embeds
            inputs_embeds = torch.cat((inputs_embeds, under_embeds), dim=1)
            selfcheck_ids = vl_chat_processor.tokenizer.encode(prompt[0])[1:]
            selfcheck_ids = torch.LongTensor(selfcheck_ids)
            selfcheck_tokens = torch.zeros((parallel_size, len(selfcheck_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size):
                selfcheck_tokens[i, :] = selfcheck_ids
            selfcheck_embeds = self.language_model.get_input_embeddings()(selfcheck_tokens)
            inputs_embeds = torch.cat((inputs_embeds, selfcheck_embeds), dim=1)
            reflect_tokens = torch.zeros((parallel_size, max_reflect_len), dtype=torch.int).cuda()
            reflect_len = 0
            eos_list = torch.zeros((parallel_size, 1), dtype=torch.int).cuda()
            add_padding = torch.zeros((parallel_size, 1), dtype=torch.int).cuda()
            eos_token = vl_chat_processor.tokenizer.encode("<｜end▁of▁sentence｜>")[-1]
            padding_token = vl_chat_processor.tokenizer.encode("<｜▁pad▁｜>")[-1]
            yes_token = vl_chat_processor.tokenizer.encode("Yes")[-1]
            no_token = vl_chat_processor.tokenizer.encode("No")[-1]
            attn_mask = torch.ones((parallel_size, inputs_embeds.shape[1]), dtype=torch.int).cuda()
            embeds_2 = inputs_embeds
            attention_mask_2 = attn_mask
            yes_list = torch.zeros((parallel_size), dtype=torch.int).cuda()
            for i in range(max_reflect_len):
                outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                logits = outputs.logits
                logits = logits[:,-1,:]
                if i == 0:
                    allowed_tokens = [yes_token, no_token]
                    allowed_tokens_logits = logits[:,allowed_tokens]
                    logits[:,:] = -math.inf
                    logits[:,allowed_tokens] = allowed_tokens_logits

                if txt_top_k:
                    v, _ = torch.topk(logits, min(txt_top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits / temperature, dim=-1)
                if txt_top_p:
                    probs_sort, probs_idx = torch.sort(probs,
                                                    dim=-1,
                                                    descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    mask = probs_sum - probs_sort > txt_top_p
                    probs_sort[mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1) 
                if i >= 1:
                    add_padding = ((reflect_tokens[:, i-1] == eos_token) | (reflect_tokens[:, i-1] == padding_token)).unsqueeze(1).to(torch.int)
                next_token = add_padding*padding_token + (1-add_padding)*next_token
                if i == 0:
                    yes_list = (next_token == yes_token)
                reflect_tokens[:, i] = next_token.squeeze(dim=-1)
                is_eos = (next_token == eos_token) 
                eos_list = eos_list | is_eos.to(torch.int)
                new_attn = 1-add_padding
                new_attn = new_attn & (~is_eos)
                attn_mask = torch.cat((attn_mask, new_attn), dim=1)
                inputs_embeds = self.language_model.get_input_embeddings()(next_token)
                reflect_len = i
                if eos_list.all():
                    break
            reflect_tokens = reflect_tokens[:,:reflect_len+1]    
            max_relect_len = reflect_len+1
            output_text_ids = reflect_tokens
            attention_mask_txt = torch.ones_like(output_text_ids).cuda()
            attention_mask_txt[output_text_ids == padding_token] = 0
            attention_mask_txt[output_text_ids == eos_token] = 0
            selfcheck = yes_list.bool()

        if 3 <= task_list[-1]:
            tokens = torch.repeat_interleave(input_ids,2,dim=0)
            for i in range(tokens.size(0)):
                if i % 2 != 0:
                    pad_list = torch.where(tokens[i]==vl_chat_processor.pad_id)[0]
                    if pad_list.shape[0]==0:
                        st = 1
                    else:
                        st = pad_list[-1].item()+2
                    tokens[i, st:-1] = vl_chat_processor.pad_id
            inputs_embeds = self.language_model.get_input_embeddings()(tokens)
            gen_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            gen_embeds_list = []  
            for i in range(len(all_imgs_1)):
                img = gen_transform(all_imgs_1[i])  
                img = img.unsqueeze(0).to(torch.bfloat16).cuda() 
                _, _, all_image_ids = self.gen_vision_model.encode(img)
                image_ids = all_image_ids[2]
                embed = self.gen_aligner(self.gen_embed(image_ids)) 
                gen_embeds_list.append(embed)
                gen_embeds_list.append(embed)
            gen_embeds = torch.cat(gen_embeds_list, dim=0)
            inputs_embeds = torch.cat((inputs_embeds, gen_embeds), dim=1)
            selfcheck_ids = vl_chat_processor.tokenizer.encode(prompt[0])[1:]
            selfcheck_ids = torch.LongTensor(selfcheck_ids)
            selfcheck_tokens = torch.zeros((2*parallel_size, len(selfcheck_ids)), dtype=torch.int).cuda()
            for i in range(2*parallel_size):
                selfcheck_tokens[i, :] = selfcheck_ids
            selfcheck_embeds = self.language_model.get_input_embeddings()(selfcheck_tokens)
            inputs_embeds = torch.cat((inputs_embeds, selfcheck_embeds), dim=1)
            attn_mask = torch.ones((2*parallel_size, inputs_embeds.shape[1]), dtype=torch.int).cuda()
            reflect_embeds = torch.ones((2*parallel_size, max_relect_len), dtype=torch.int).cuda()
            for i in range(2*parallel_size):
                reflect_embeds[i] = output_text_ids[i//2]
            new_attn = torch.ones((2*parallel_size, max_relect_len), dtype=torch.int).cuda()
            for i in range(2*parallel_size):
                new_attn[i] = attention_mask_txt[i//2]
            reflect_embeds = self.language_model.get_input_embeddings()(reflect_embeds)
            inputs_embeds = torch.cat((inputs_embeds, reflect_embeds), dim=1)
            attn_mask = torch.cat((attn_mask, new_attn), dim=1)
            regen_ids = vl_chat_processor.tokenizer.encode(prompt[1])[1:]
            regen_ids = torch.LongTensor(regen_ids)
            regen_tokens = torch.zeros((2*parallel_size, len(regen_ids)), dtype=torch.int).cuda()
            for i in range(2*parallel_size):
                regen_tokens[i, :] = regen_ids
            regen_embeds = self.language_model.get_input_embeddings()(regen_tokens)
            inputs_embeds = torch.cat((inputs_embeds, regen_embeds), dim=1)
            new_attn = torch.ones((2*parallel_size, regen_ids.shape[0]), dtype=torch.int).cuda()
            attn_mask = torch.cat((attn_mask, new_attn), dim=1)
            embeds_3 = inputs_embeds
            attention_mask_3 = attn_mask
            new_generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
            for i in tqdm(range(image_token_num_per_image)):
                outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state
                new_attn = torch.ones((2*parallel_size, 1), dtype=torch.int).cuda()
                attn_mask = torch.cat((attn_mask, new_attn), dim=1)
                logits = self.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                if img_top_k:
                    v, _ = torch.topk(logits, min(img_top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits / temperature, dim=-1)
                if img_top_p:
                    probs_sort, probs_idx = torch.sort(probs,
                                                    dim=-1,
                                                    descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    mask = probs_sum - probs_sort > img_top_p
                    probs_sort[mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                new_generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
            new_dec = self.gen_vision_model.decode_code(new_generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
            new_dec = new_dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            new_dec = np.clip((new_dec + 1) / 2 * 255, 0, 255)
            new_visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            new_visual_img[:, :, :] = new_dec
            for i in range(parallel_size):
                all_imgs_2.append(PIL.Image.fromarray(new_visual_img[i]))
            img_ids_2 = new_generated_tokens
        if img_path:
            if dist.get_rank() % torch.cuda.device_count() == 0:
                os.makedirs(img_path, exist_ok=True)
                final_visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
                for i in range(parallel_size):
                    if selfcheck[i]==True:
                        final_visual_img[i,:,:] = visual_img[i,:,:]
                    else:
                        final_visual_img[i,:,:] = new_visual_img[i,:,:]
                self.save_stack_images(visual_img, batch_size=visual_img.shape[0], save_path=os.path.join(img_path, f'{cur_step}_first.png'))
                self.save_stack_images(new_visual_img, batch_size=new_visual_img.shape[0], save_path=os.path.join(img_path, f'{cur_step}_second.png'))
                self.save_stack_images(final_visual_img, batch_size=final_visual_img.shape[0], save_path=os.path.join(img_path, f'{cur_step}_final.png'))
                
        return (img_ids_1, all_imgs_1), (img_ids_2, all_imgs_2), (output_text_ids, selfcheck.squeeze(), attention_mask_txt), (embeds_1, attention_mask_1), (embeds_2, attention_mask_2), (embeds_3, attention_mask_3)

    @torch.no_grad()
    def edit_image(
        self,
        vl_chat_processor: VLChatProcessor,
        input_ids,
        attention_mask,
        image1,
        image_seq_mask,
        st=0,
        ed=0,
        cur_step=0,
        temperature: float = 1,
        parallel_size: int = 16,
        cfg_weight: float = 5,
        set_cfg=True,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        parallel_size = input_ids.shape[0] // 2

        image_embeds, _ = self.prepare_embedding(image1)
        input_ids[input_ids < 0] = 0 
        tokens = input_ids

        inputs_embeds = self.language_model.get_input_embeddings()(tokens)
        
        for i in range(inputs_embeds.shape[0]):
            inputs_embeds[i][image_seq_mask[i]] = image_embeds[i]
        
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        B = attention_mask.shape[0]
        from tqdm import tqdm
        for i in tqdm(range(image_token_num_per_image)):
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.gen_head(hidden_states[:, -1, :])
            if set_cfg:
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
            
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            if set_cfg:
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(B, 1).to(attention_mask)], dim=1)
        
        dec = self.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        final_imgs = [Image.fromarray(img) for img in visual_img]

        return generated_tokens, final_imgs, (tokens, attention_mask)

    def forward(
        self, text_inputs_ids, img_ids, attention_mask, logits_to_keep=0, addcfg=True, guidance_scale=5.0
    ):
        inputs_embeds = self.language_model.get_input_embeddings()(text_inputs_ids)
        if img_ids.shape[0] < text_inputs_ids.shape[0]:
            new_img_ids = torch.repeat_interleave(img_ids, 2, dim=0) 
        else:
            new_img_ids = img_ids
        
        visual_embeds = self.gen_aligner(self.gen_embed(new_img_ids))
        inputs_embeds = torch.cat([inputs_embeds, visual_embeds], dim=1)
        
        if addcfg == False:
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B, L, V)
            hidden_states = outputs.last_hidden_state
            logits = self.gen_head(hidden_states)
            logits = logits[:, -1-logits_to_keep:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = img_ids.long()  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        else:
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B, L, V)
            hidden_states = outputs.last_hidden_state
            print(inputs_embeds.size(), hidden_states.size())
            logits = self.gen_head(hidden_states)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_cond - (guidance_scale-1) / guidance_scale *logit_uncond
            logits = logits[:, -1-logits_to_keep:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = img_ids.long()
        print(logits.size(), input_ids.size())

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1) # index 为什么是input_ids_row？，logits和input_ids的顺序不应该是对应的？
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
