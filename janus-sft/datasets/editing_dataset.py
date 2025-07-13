import os
import json
import torch
import random
import linecache
import numpy as np
from PIL import Image
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

class ImageEditingDataset(Dataset):
    def __init__(
        self,
        model_path
    ):  
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.num1, self.num2 = 0, 0
        fn = []
        self.alldata = []
        alldirs = [
            'data/editing_examples/metadata/', 
            # ...
        ]
        for i in alldirs:
          allfs = os.listdir(i)
          for f in allfs:
            fn.append(os.path.join(i, f))
        for f in fn:
          curnum = len(linecache.getlines(f))
          for i in range(curnum):
            try:
              curcontent = json.loads(linecache.getline(f, i+1).strip())
              if 0==0: 
                self.alldata.append(curcontent)
            except Exception as e:
              pass

        print('data num:', len(self.alldata))
        self.num_gen_image_tokens = 576
        self.generate = ' Please generate an image.'
        p1 = '<|User|>: <begin_of_image>' + self.vl_chat_processor.image_tag*self.num_gen_image_tokens + self.vl_chat_processor.image_end_tag
        p2 = ' Please generate an image.\n\n<|Assistant|>:<begin_of_image>'
        self.input_ids1 = self.tokenizer.encode(p1, return_tensors='pt').squeeze(0) 
        self.input_ids2 = self.tokenizer.encode(p2, return_tensors='pt', add_special_tokens=False).squeeze(0)
        self.max_prompt_length = 90
        
    def __getitem__(self, idx):
        while True:
          try:
            curdata = self.alldata[idx]
            image1 = Image.open(curdata['input']).convert('RGB')
            image1 = self.gen_transform(image1)
            image2 = Image.open(curdata['output']).convert('RGB')
            image2 = self.gen_transform(image2)
            prompt = curdata['text']
            break
          except Exception as e:
            print(e)
            idx = random.randint(0, len(self.alldata)-1)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False, max_length=self.max_prompt_length, truncation=True).squeeze(0)
        if random.random() <= 0.1:
            input_ids[:] = self.vl_chat_processor.pad_id
        input_ids = torch.cat((self.input_ids1, input_ids, self.input_ids2), dim=0)

        return {"input_ids":input_ids, "image1":image1, "image2":image2, "task_type":2}

    def __len__(self):
        return len(self.alldata)

def my_collate_fn(batch):
    return batch

def ImageEditingDataloader(cfg):
    dataset = ImageEditingDataset(model_path=cfg.model.processor_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    return loader
