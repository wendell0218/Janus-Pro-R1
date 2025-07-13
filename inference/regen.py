import os
import json
import torch
import linecache
import PIL.Image
import numpy as np
from typing import List
from torchvision import transforms
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: List[str],
    gen_path: str,
    reason_path: str,
    temperature: float = 1.0,
    cfg_weight: float = 5,
    parallel_size: int = 4,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    img_top_k: int = 1,
    img_top_p: float = 1.0,
):
    images, image_ids, output_text_ids, selfcheck, regen_images = [], [], [], [], []
    input_ids = vl_chat_processor.tokenizer.encode(prompt[0])
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens) 
    embeds_1 = inputs_embeds
    gen_res = os.listdir(gen_path)
    gen_res = [os.path.join(gen_path, i) for i in gen_res]
    for img_path in gen_res:
        img = PIL.Image.open(img_path).convert("RGB")
        images.append(img)
        gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        tr_img = gen_transform(img)  
        tr_img = tr_img.unsqueeze(0).to(torch.bfloat16).cuda() 
        _, _, all_image_ids = mmgpt.gen_vision_model.encode(tr_img)
        image_ids.append(all_image_ids[2])
    inputs_embeds = embeds_1[::2,:,:] 
    under_embeds = torch.zeros((parallel_size, image_token_num_per_image, 4096), dtype=torch.bfloat16).cuda()
    for i in range(parallel_size):
        img_prompt = "<image_placeholder>"
        prepare_inputs = vl_chat_processor(
            prompt=img_prompt, images=[images[i]], force_batchify=True
        ).to(vl_gpt.device)
        img_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs) 
        img_embeds = img_embeds[:,2:-1,:] 
        under_embeds[i,:,:] = img_embeds
    inputs_embeds = torch.cat((inputs_embeds, under_embeds), dim=1)
    selfcheck_ids = vl_chat_processor.tokenizer.encode(prompt[1])[1:]
    selfcheck_ids = torch.LongTensor(selfcheck_ids)
    selfcheck_tokens = torch.zeros((parallel_size, len(selfcheck_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size):
        selfcheck_tokens[i, :] = selfcheck_ids
    selfcheck_embeds = mmgpt.language_model.get_input_embeddings()(selfcheck_tokens)
    inputs_embeds = torch.cat((inputs_embeds, selfcheck_embeds), dim=1)
    padding_token = vl_chat_processor.tokenizer.encode("<｜▁pad▁｜>")[-1]
    attn_mask = torch.ones((parallel_size, inputs_embeds.shape[1]), dtype=torch.int).cuda()
    max_relect_len = 0
    reason_res = linecache.getlines(reason_path)
    reason_res = [json.loads(i)["reason"] for i in reason_res]
    for reflect in reason_res:
        if "Yes" in reflect:
            selfcheck.append(1)
        else:
            selfcheck.append(0)
        tokens = torch.LongTensor(vl_chat_processor.tokenizer.encode(reflect)[1:])
        output_text_ids.append(tokens)
        max_relect_len = max(max_relect_len, tokens.shape[0])
    for i in range(len(output_text_ids)):
        padding = torch.tensor([padding_token]*(max_relect_len-output_text_ids[i].shape[0]))
        output_text_ids[i] = torch.cat((output_text_ids[i], padding), dim=0)
    output_text_ids = torch.stack(output_text_ids, dim=0).to(dtype=torch.long).cuda()
    attention_mask_txt = torch.ones_like(output_text_ids).cuda()
    attention_mask_txt[output_text_ids == padding_token] = 0
    input_ids = vl_chat_processor.tokenizer.encode(prompt[0])
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    gen_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    gen_embeds_list = []  
    for i in range(len(images)):
        img = gen_transform(images[i])  
        img = img.unsqueeze(0).to(torch.bfloat16).cuda() 
        _, _, all_image_ids = mmgpt.gen_vision_model.encode(img)
        image_ids = all_image_ids[2]
        embed = mmgpt.gen_aligner(mmgpt.gen_embed(image_ids)) 
        gen_embeds_list.append(embed)
        gen_embeds_list.append(embed)
    gen_embeds = torch.stack(gen_embeds_list, dim=0)
    inputs_embeds = torch.cat((inputs_embeds, gen_embeds), dim=1)
    selfcheck_ids = vl_chat_processor.tokenizer.encode(prompt[1])[1:]
    selfcheck_ids = torch.LongTensor(selfcheck_ids)
    selfcheck_tokens = torch.zeros((2*parallel_size, len(selfcheck_ids)), dtype=torch.int).cuda()
    for i in range(2*parallel_size):
        selfcheck_tokens[i, :] = selfcheck_ids
    selfcheck_embeds = mmgpt.language_model.get_input_embeddings()(selfcheck_tokens)
    inputs_embeds = torch.cat((inputs_embeds, selfcheck_embeds), dim=1)
    attn_mask = torch.ones((2*parallel_size, inputs_embeds.shape[1]), dtype=torch.int).cuda()
    reflect_embeds = torch.ones((2*parallel_size, max_relect_len), dtype=torch.int).cuda()
    for i in range(2*parallel_size):
        reflect_embeds[i] = output_text_ids[i//2]
    new_attn = torch.ones((2*parallel_size, max_relect_len), dtype=torch.int).cuda()
    for i in range(2*parallel_size):
        new_attn[i] = attention_mask_txt[i//2]
    reflect_embeds = mmgpt.language_model.get_input_embeddings()(reflect_embeds)
    inputs_embeds = torch.cat((inputs_embeds, reflect_embeds), dim=1)
    attn_mask = torch.cat((attn_mask, new_attn), dim=1)
    regen_ids = vl_chat_processor.tokenizer.encode(prompt[2])[1:]
    regen_ids = torch.LongTensor(regen_ids)
    regen_tokens = torch.zeros((2*parallel_size, len(regen_ids)), dtype=torch.int).cuda()
    for i in range(2*parallel_size):
        regen_tokens[i, :] = regen_ids
    regen_embeds = mmgpt.language_model.get_input_embeddings()(regen_tokens)
    inputs_embeds = torch.cat((inputs_embeds, regen_embeds), dim=1)
    new_attn = torch.ones((2*parallel_size, regen_ids.shape[0]), dtype=torch.int).cuda()
    attn_mask = torch.cat((attn_mask, new_attn), dim=1)
    new_generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        new_attn = torch.ones((2*parallel_size, 1), dtype=torch.int).cuda()
        attn_mask = torch.cat((attn_mask, new_attn), dim=1)
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
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
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    new_dec = mmgpt.gen_vision_model.decode_code(new_generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    new_dec = new_dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    new_dec = np.clip((new_dec + 1) / 2 * 255, 0, 255)
    new_visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    new_visual_img[:, :, :] = new_dec
    for i in range(parallel_size):
        regen_images.append(PIL.Image.fromarray(new_visual_img[i]))

    return images, regen_images, selfcheck

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--caption", type=str, default="a brown giraffe and a white stop sign")
    parser.add_argument("--gen_path", type=str, default="results/samples")
    parser.add_argument("--reason_path", type=str, default='results/reason.jsonl')
    parser.add_argument("--regen_path", type=str, default='results/regen_samples')
    parser.add_argument("--cfg", type=float, default=5.0)

    args = parser.parse_args()

    parallel_size = len(os.listdir(args.gen_path))
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    if args.ckpt_path is not None:
        state_dict = torch.load(f"{args.ckpt_path}", map_location="cpu")
        vl_gpt.load_state_dict(state_dict)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    prompt = [
        f'<|User|>: {args.caption}\n\n<|Assistant|>:<begin_of_image>',
        '<end_of_image>\nLet me think Does this image match the prompt...',
        '<｜end▁of▁sentence｜>\nNext, I will draw a new image<begin_of_image>'
    ]
    images, regen_images, selfcheck = generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        parallel_size = parallel_size,
        gen_path=args.gen_path,
        reason_path=args.reason_path,
        cfg_weight=args.cfg,
    )
    os.makedirs(args.regen_path, exist_ok=True)
    for i in range(parallel_size):
        img_name = str(i).zfill(4)+".png"
        save_path = os.path.join(args.regen_path, img_name)
        if selfcheck[i]:
            images[i].save(save_path)
        else:
            regen_images[i].save(save_path)