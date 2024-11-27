import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils,  io
from torch import nn

import cv2 
from sklearn import preprocessing
import rasterio as rio
import numpy as np
from PIL import Image
import random, os
import clip
from torchvision import utils
from tqdm import tqdm


class SenDataset(Dataset):
    def __init__(self, root_dir, emb_path, sen_image_path, bands=None, Image_Size=224):
        super().__init__()
        if bands is None:
            bands = [3,2,1]
        self.paths_sen = np.load(sen_image_path)
        self.emb_dict = torch.load(emb_path)#, map_location= 'cpu'
        self.root_dir = root_dir
        self.bands = bands
        print(f'{len(self.paths_sen)} sentinel images found')
       

    def _transform(self, image):
        transform = transforms.Compose([
            #transforms.ToPILImage(mode='RGB'),
            transforms.CenterCrop((32, 32)),
            transforms.Resize((224,224), interpolation= transforms.InterpolationMode.BILINEAR),
            transforms.RandAugment(3,5,interpolation= transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.347, 0.376, 0.296), (0.269, 0.261, 0.276)),
        ])
        return transform(image)
    
    def _load_image(self, path):
        with rio.open(path) as src:
            img = src.read(self.bands)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)#
        img = np.uint8(img).transpose(1, 2, 0)
        img = Image.fromarray(img, mode='RGB')#
        return img
    
    def __len__(self):
        return len(self.paths_sen)

    def __getitem__(self, index):
        path_sen = self.paths_sen[index]
        
        id_sen = f'{path_sen.split("/")[0]}_{path_sen.split("/")[-1].split(".")[0].split("_")[0]}'
        w_emb = self.emb_dict[f'{id_sen}W']
        e_emb = self.emb_dict[f'{id_sen}E']
        n_emb = self.emb_dict[f'{id_sen}N']
        s_emb = self.emb_dict[f'{id_sen}S']
        
        frozen_emb = torch.cat((w_emb, e_emb, n_emb, s_emb), dim=0)

        img_positive = self._load_image(self.root_dir+path_sen)
        img_positive = self._transform(img_positive)

       
        return frozen_emb, img_positive
    
class TestDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.embeddings = torch.randn(10, 4, 1024)  # Example data
            self.pos_images = torch.randn(10, 3, 224, 224)

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.pos_images[idx]