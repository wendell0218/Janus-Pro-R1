import os
import math
import json
import torch
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
    temperature: float = 1.0,
    parallel_size: int = 4,
    image_token_num_per_image: int = 576,
    max_reflect_len: int = 256,
    txt_top_k: int = 50,
    txt_top_p: float = 1.0,
):
    images, image_ids = [], []
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
    reflect_tokens = torch.zeros((parallel_size, max_reflect_len), dtype=torch.int).cuda()
    reflect_len = 0
    eos_list = torch.zeros((parallel_size, 1), dtype=torch.int).cuda()
    add_padding = torch.zeros((parallel_size, 1), dtype=torch.int).cuda()
    eos_token = vl_chat_processor.tokenizer.encode("<｜end▁of▁sentence｜>")[-1]
    padding_token = vl_chat_processor.tokenizer.encode("<｜▁pad▁｜>")[-1]
    yes_token = vl_chat_processor.tokenizer.encode("Yes")[-1]
    no_token = vl_chat_processor.tokenizer.encode("No")[-1]
    attn_mask = torch.ones((parallel_size, inputs_embeds.shape[1]), dtype=torch.int).cuda()
    yes_list = torch.zeros((parallel_size), dtype=torch.int).cuda()
    for i in range(max_reflect_len):
        outputs = mmgpt.language_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
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
        inputs_embeds = mmgpt.language_model.get_input_embeddings()(next_token) 
        reflect_len = i
        if eos_list.all():
            break
    reflect_tokens = reflect_tokens[:,:reflect_len+1]    
    output_text_ids = reflect_tokens
    attention_mask_txt = torch.ones_like(output_text_ids).cuda()
    attention_mask_txt[output_text_ids == padding_token] = 0
    attention_mask_txt[output_text_ids == eos_token] = 0
    selfcheck = yes_list.cpu().tolist()
    selfcheck = [int(item[0]) for item in selfcheck]

    return output_text_ids, selfcheck

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--caption", type=str, default="a brown giraffe and a white stop sign")
    parser.add_argument("--gen_path", type=str, default="results/samples")
    parser.add_argument("--reason_path", type=str, default='results/reason.jsonl')

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
    ]
    output_text_ids, selfcheck = generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        parallel_size = parallel_size,
        gen_path=args.gen_path,
    )
    os.makedirs(os.path.dirname(args.reason_path), exist_ok=True)
    with open(args.reason_path, 'w') as f:
        for i in range(parallel_size):
            reason_data = {"prompt": args.caption}
            img_name = str(i).zfill(4)
            reason_data["filename"] = os.path.join(args.gen_path, f"{img_name}.png")
            reason_data["correct"] = bool(selfcheck[i])
            reason_data["reason"] = vl_chat_processor.tokenizer.decode(output_text_ids[i].cpu().tolist(), skip_special_tokens=True)
            reason_data = json.dumps(reason_data, ensure_ascii=False)
            f.write(reason_data+'\n')
