import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob

class CVACTDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']
        ids = ids[anuData[f'{split}Set'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
   
            # if not os.path.exists(grd_path) or not os.path.exists(sat_path):
            self.idx_ignor.add(idx)
            # else:
            self.idx2label[idx] = i
            ids_list.append(idx)
            i+=1
        
        #print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list


    def __getitem__(self, index):
        
        idx = self.samples[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
        elif self.img_type == "query_BEV":
            path = f'{self.data_folder}/ANU_data_small/bev/{idx}_grdView.jpg' 

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)

            
