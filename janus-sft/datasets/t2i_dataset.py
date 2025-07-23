import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import linecache
from collections import defaultdict
import json
from torchvision import transforms
from models import VLChatProcessor
from torch.utils.data import Dataset, DataLoader

def center_crop_arr(pil_image, image_size):
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

class TextToImageDataset(Dataset):
    def __init__(
        self,
        task_type,
        model_path,
    ):
        self.task_type = task_type
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.selfcheck_text = '<end_of_image>\nLet me think Does this image match the prompt...'
        self.data = []
        self.internvl_data = []
        self.num_gen_image_tokens = 576
        self.max_reason_length = 184
        self.max_prompt_length = 200
        self.internvl_len = 0
        self.load_from_internvl()

    def load_from_internvl(self):
        self.internvl_data = []
        if self.task_type == 0:
            self.thre = 0.7
        elif self.task_type == 1:
            self.thre = 0.7
        elif self.task_type == 2:
            self.thre = 0.7
        paths = [
            "data/t2i_examples/"
        ]
        if self.task_type == 0:
            imageid = 'flux'
            for root_path in paths:
                label_path = os.path.join(root_path, 'labels', f'{imageid}_{self.thre}.jsonl')
                num = len(linecache.getlines(label_path))
                for i in range(num):
                    curcontent = json.loads(linecache.getline(label_path, i+1))
                    prompt = curcontent['prompt']
                    curdatalist = curcontent['data']
                    for curdata in curdatalist:
                        if curdata['state'] == 4:
                            self.internvl_data.append((prompt, curdata['img_path']))
            self.internvl_len = len(self.internvl_data)
            self.data.extend(self.internvl_data)
        
        if self.task_type == 1:
            allimageids = [
                'flux',
                # ...
            ]
            data1 = []
            data2 = []
            for imageid in allimageids:
                for root_path in paths:
                    label_path = os.path.join(root_path, 'labels', f'{imageid}_{self.thre}.jsonl')
                    num = len(linecache.getlines(label_path))
                    for i in range(num):
                        curcontent = json.loads(linecache.getline(label_path, i+1))
                        prompt = curcontent['prompt']
                        curdatalist = curcontent['data']
                        for curdata in curdatalist:
                            if curdata['state'] == 4:
                                myreason = 'Yes'+curdata['reason'][3:]
                                data1.append((prompt, curdata['img_path'], curdata['state'], myreason))
                            if curdata['state'] == 0 or curdata['state'] == 1:
                                myreason = 'No'+curdata['reason'][2:]
                                data2.append((prompt, curdata['img_path'], curdata['state'], myreason))
            len1 = len(data1)
            len2 = len(data2)
            if len1>len2:
                data1 = random.sample(data1, len2)
            else:
                data2 = random.sample(data2, len1)
            self.internvl_data.extend(data1)
            self.internvl_data.extend(data2)
            self.internvl_len = len(self.internvl_data)
            self.data.extend(self.internvl_data)
        
        if self.task_type == 2:
            both_prompt = defaultdict(dict)
            for j, root_path in enumerate(paths):
                imageid = 'flux'
                label_path = os.path.join(root_path, 'labels', f'{imageid}_{self.thre}.jsonl')
                num = len(linecache.getlines(label_path))
                for idx in range(num):
                    curcontent = json.loads(linecache.getline(label_path, idx+1))
                    prompt = curcontent['prompt']
                    curdatalist = curcontent['data']
                    pid = curcontent['promptid']
                    pid = f'{j}_{pid}'
                    flag = False
                    for curdata in curdatalist:
                        if curdata['state'] == 4:
                            flag = True
                            if pid not in both_prompt.keys():
                                both_prompt[pid] = {'prompt':prompt, 'pid':pid, 'pos':[], 'neg':[]}
                            both_prompt[pid]['pos'].append((curdata['img_path'], curdata['state'], curdata['reason'], curdata['prob']))
                        elif curdata['state'] == 0 or curdata['state'] == 1:
                            flag = True
                            if pid not in both_prompt.keys():
                                both_prompt[pid] = {'prompt':prompt, 'pid':pid, 'pos':[], 'neg':[]}
                            both_prompt[pid]['neg'].append((curdata['img_path'], curdata['state'], curdata['reason'], curdata['prob']))
                            
                    if flag:
                        if len(both_prompt[pid]['pos']) == 0:
                            del both_prompt[pid]
                allimageids = [
                    'janus',
                    # ...
                ]
                for imageid in allimageids:
                    label_path = os.path.join(root_path, 'labels', f'{imageid}_{self.thre}.jsonl')
                    num = len(linecache.getlines(label_path))
                    for idx in range(num):
                        curcontent = json.loads(linecache.getline(label_path, idx+1))
                        prompt = curcontent['prompt']
                        curdatalist = curcontent['data']
                        pid = curcontent['promptid']
                        if pid not in both_prompt.keys():
                            continue
                        for curdata in curdatalist:
                            if curdata['state'] == 0 or curdata['state'] == 1:
                                both_prompt[pid]['neg'].append((curdata['img_path'], curdata['state'], curdata['reason'], curdata['prob']))
                                
            all_pid = both_prompt.keys()
            self.internvl_allids = []
            for pid in all_pid:
                pos_list = both_prompt[pid]['pos']
                neg_list = both_prompt[pid]['neg']
                if len(pos_list)==0 or len(neg_list)==0:
                    continue
                sorted_pos_list = sorted(pos_list, key=lambda x:x[3])
                sorted_neg_list = sorted(neg_list, key=lambda x:x[3])
                if pos_list[-1][3] <= neg_list[0][3]:
                    continue
                else:
                    both_prompt[pid]['pos'] = sorted_pos_list
                    both_prompt[pid]['neg'] = sorted_neg_list
                self.internvl_allids.append(pid)
            
            self.internvl_data = both_prompt
            self.internvl_len = len(self.internvl_allids)

    def __getitem__(self, idx):
        if self.task_type==0:
            curdata = self.data[idx]
            image = Image.open(curdata[1]).convert('RGB')
            image = self.gen_transform(image)
            conversation = [
                {
                    "role": "<|User|>",
                    "content":curdata[0],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_prompt_length, truncation=True).squeeze(0)
            if random.random() < 0.1:
                input_ids[1:-1] = self.vl_chat_processor.pad_id

            return {"input_ids": input_ids, "image": image, "task_type": 0}
    
        elif self.task_type==1:
            curdata = self.data[idx]
            image = Image.open(curdata[1]).convert('RGB')
            image = self.vl_chat_processor.image_processor([image])['pixel_values'].squeeze(0)
            all_input_ids,all_labels = [],[]
            conversation = [
                {
                    "role": "<|User|>",
                    "content":curdata[0],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            input_ids1 = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_prompt_length, truncation=True).squeeze(0)
            labels1 = torch.ones(input_ids1.shape, dtype=input_ids1.dtype) * -100
            all_input_ids.append(input_ids1)
            all_labels.append(labels1)
            all_input_ids.append(self.vl_chat_processor.image_id * torch.ones((self.vl_chat_processor.num_image_tokens,), dtype=torch.long))
            all_labels.append(-100 * torch.ones((self.vl_chat_processor.num_image_tokens,), dtype=torch.long))
            selfcheck_text = self.selfcheck_text
            input_ids2 = self.tokenizer.encode(selfcheck_text, add_special_tokens=False, return_tensors='pt').squeeze(0)
            labels2 = torch.ones(input_ids2.shape, dtype=input_ids2.dtype) * -100
            all_input_ids.append(input_ids2)
            all_labels.append(labels2)
            answer = curdata[3]
            answer += self.tokenizer.eos_token
            labels = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors='pt', max_length=self.max_reason_length, truncation=True).squeeze(0)
            all_input_ids.append(labels)
            all_labels.append(labels)
            input_ids = torch.cat(all_input_ids,dim=0)
            labels = torch.cat(all_labels, dim=0)
            return {"input_ids":input_ids, "image":image, "labels": labels, "task_type":1}
        else:
            allids = self.internvl_allids
            alldata = self.internvl_data
            curidx = idx
            pid = allids[curidx]
            curdata = alldata[pid]
            prompt = curdata['prompt']
            curtry = 0
            while True:
                curtry += 1
                inc_data = random.choice(curdata['neg'])
                cor_data = random.choice(curdata['pos'])
                if cor_data[3] > inc_data[3]:
                    break
                elif curtry>=3:
                    inc_data = curdata['neg'][0]
                    cor_data = curdata['pos'][-1]
                    break
            all_input_ids = []
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            input_ids1 = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_prompt_length, truncation=True).squeeze(0)
            if random.random() < 0.1:
                input_ids1[1:-1] = self.vl_chat_processor.pad_id
            all_input_ids.append(input_ids1)
            all_input_ids.append(self.vl_chat_processor.image_id * torch.ones((self.num_gen_image_tokens,), dtype=torch.long))
            selfcheck_text = self.selfcheck_text
            selfcheck_text += inc_data[2]
            selfcheck_text += self.tokenizer.eos_token
            selfcheck_text += '\nNext, I will draw a new image'+ self.vl_chat_processor.image_start_tag
            input_ids2 = self.tokenizer.encode(selfcheck_text, add_special_tokens=False, return_tensors='pt', max_length=self.max_reason_length+16, truncation=True).squeeze(0)
            all_input_ids.append(input_ids2)
            input_ids = torch.cat(all_input_ids,dim=0)
            image1 = self.gen_transform(Image.open(inc_data[0]).convert('RGB'))
            image2 = self.gen_transform(Image.open(cor_data[0]).convert('RGB'))

            return {"input_ids":input_ids, "image1":image1, "image2":image2, "task_type":2}

    def __len__(self):
        return self.internvl_len

def my_collate_fn(batch):
    return batch

def TextToImageDataloader(cfg, tasks=[0,1,2]):
    probs = []
    dataloaders = []
    if 0 in tasks:
        dataset1 = TextToImageDataset(
            task_type=0,
            model_path=cfg.model.processor_path,
        )
        sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1)
        loader1 = DataLoader(
            dataset1,
            batch_size=cfg.dataloader.train.task1.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            sampler=sampler1,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            drop_last=True,
            collate_fn=my_collate_fn
        )
        dataloaders.append(loader1)
        probs.append(cfg.dataloader.train.task1.sample_ratio)
    
    if 1 in tasks:
        dataset2 = TextToImageDataset(
            task_type=1,
            model_path=cfg.model.processor_path,
        )
        sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2)
        loader2 = DataLoader(
            dataset2,
            batch_size=cfg.dataloader.train.task2.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            sampler=sampler2,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            drop_last=True,
            collate_fn=my_collate_fn
        )
        dataloaders.append(loader2)
        probs.append(cfg.dataloader.train.task2.sample_ratio)
    
    if 2 in tasks:
        dataset3 = TextToImageDataset(
            task_type=2,
            model_path=cfg.model.processor_path,
        )
        sampler3 = torch.utils.data.distributed.DistributedSampler(dataset3)
        loader3 = DataLoader(
            dataset3,
            batch_size=cfg.dataloader.train.task3.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            sampler=sampler3,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            drop_last=True,
            collate_fn=my_collate_fn
        )
        dataloaders.append(loader3)
        probs.append(cfg.dataloader.train.task3.sample_ratio)

    probs = [p / sum(probs) for p in probs]
    if len(dataloaders)==1:
        return dataloaders[0], probs[0]

    return dataloaders, probs
