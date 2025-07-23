import os
import json
import linecache
import numpy as np
import pandas as pd
from myutils import mask_decode
from torch.utils.data import Dataset

class MyT2IDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.prompt_list = []
        
        num = len(linecache.getlines(data_path))

        for i in range(num):
            curcontent = linecache.getline(data_path, i+1)
            self.prompt_list.append(curcontent.strip())
        
    def __getitem__(self, idx):
       
        return {
            'text':self.prompt_list[idx],
        }
    
    def __len__(self):
        return len(self.prompt_list)

class MyEditDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_list = []
        
        num = len(linecache.getlines(data_path))

        for i in range(num):
            curcontent = linecache.getline(data_path, i+1)
            json_line = json.loads(curcontent.strip())
            self.data_list.append(json_line)
        
    def __getitem__(self, idx):
       
        return self.data_list[idx]
    
    def __len__(self):
        return len(self.data_list)