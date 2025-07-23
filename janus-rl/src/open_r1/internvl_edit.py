import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internvl_chat import InternVLChatConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image2(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

import torch.nn.functional as F
from torch import nn
class InternVLReward(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config, vision_model, language_model, use_flash_attn)

        # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        path = '/mnt/prev_nas/refine_draw_RL/models/models/OpenGVLab/InternVL2_5-8B'
        self.model = InternVLChatModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        for n,p in self.named_parameters():
            p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.vocab_dict = self.tokenizer.get_vocab()

    def evaluate(self, images, images2, prompts):
        assert len(prompts) == len(images), len(prompts)==len(images2)

        pixel_values = []
        for idx in range(len(images)):
            pixel_values.append(load_image2(images[idx], max_num=12).to(torch.bfloat16).cuda())
            pixel_values.append(load_image2(images2[idx], max_num=12).to(torch.bfloat16).cuda())

        num_patches_list = [pv.size(0) for pv in pixel_values]
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        pixel_values = torch.cat(pixel_values, dim=0)

        questions = [f'Please evaluate the following image edit based on the provided instructions:\n\nImage-1: <image>\nImage-2: <image>\nThe first image is the original image and the second image is the edited image\nEditing Instructions: {prompt}\nDoes the edited image follow the editing instructions? Please directly respond with yes or no.' for prompt in prompts]
        with torch.no_grad():
            responses = self.my_batch_chat2(self.tokenizer, pixel_values,
                                                num_patches_list=num_patches_list,
                                                questions=questions,
                                                generation_config=generation_config)


        logitsm = F.softmax(responses.scores[0], dim=-1).detach().cpu().squeeze(0)

        yes_prob = (logitsm[:, self.vocab_dict['yes']] + logitsm[:, self.vocab_dict['Yes']] + logitsm[:, self.vocab_dict['YES']])
        no_prob = (logitsm[:, self.vocab_dict['no']] + logitsm[:, self.vocab_dict['No']] + logitsm[:, self.vocab_dict['NO']])
        prob_list = yes_prob / (yes_prob + no_prob+1.0e-10)
        yes_prob_norm = [round(pp.item(), 4) for pp in prob_list]
        score1 = yes_prob_norm

        
        questions = [f'Please evaluate the following image edit based on the provided instructions:\n\nImage-1: <image>\nImage-2: <image>\nThe first image is the original image and the second image is the edited image\nEditing Instructions: {prompt}\nDoes the area of the edited image that is unrelated to the editing instructions remain consistent with the original image? Please directly respond with yes or no.' for prompt in prompts]
        with torch.no_grad():
            responses = self.my_batch_chat2(self.tokenizer, pixel_values,
                                            num_patches_list=num_patches_list,
                                            questions=questions,
                                            generation_config=generation_config)

        

        logitsm = F.softmax(responses.scores[0], dim=-1).detach().cpu().squeeze(0)

        yes_prob = (logitsm[:, self.vocab_dict['yes']] + logitsm[:, self.vocab_dict['Yes']] + logitsm[:, self.vocab_dict['YES']])
        no_prob = (logitsm[:, self.vocab_dict['no']] + logitsm[:, self.vocab_dict['No']] + logitsm[:, self.vocab_dict['NO']])
        prob_list = yes_prob / (yes_prob + no_prob+1.0e-10)
        yes_prob_norm = [round(pp.item(), 4) for pp in prob_list]
        score2 = yes_prob_norm


        return score1, score2

        
