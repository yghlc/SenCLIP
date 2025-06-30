import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils,  io
from torch import nn
import cv2 , json
from sklearn import preprocessing
import rasterio as rio
import numpy as np
from PIL import Image
import random
import clip
from torchvision import utils
from tqdm import tqdm
from torchgeo import datasets
from random import sample

class BigEarthNetDataset(Dataset):
    def __init__(self, root_dir, split='test',num_classes=19, bands=None, Image_Size=224, norm_value = 'BingCLIP', download=False):
        super().__init__()
        bing_mean, bing_std = (0.347, 0.376, 0.296), (0.269, 0.261, 0.276)
        clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.dataset = datasets.BigEarthNet(root=root_dir, 
                                        split=split, 
                                        bands='s2', 
                                        num_classes= num_classes, 
                                        transforms=None,
                                        download=download)
        
        self.class_lists = self.dataset.class_sets[num_classes]
        if norm_value in ["CLIP", "RemoteCLIP"]:   
            self.mean = clip_mean
            self.std  = clip_std  
        else: 
            self.mean = bing_mean
            self.std  = bing_std
        if bands is None:
            bands = [3,2,1]
       
        self.root_dir = root_dir

        self.bands = bands
        self.Image_Size = Image_Size
        print(f'{len(self.dataset)}  images found')
       

    def _transform(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.Image_Size, self.Image_Size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        return transform(image)
  
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image = self.dataset[index]['image']
        label = self.dataset[index]['label']
        
        image = np.asarray(image[1:4,:,:])
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)#
        image = np.uint8(image).transpose(1, 2, 0)
        image = Image.fromarray(image, mode='RGB')
        
        return self._transform(image), label
    
   
