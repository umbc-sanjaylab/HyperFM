import os
import pandas as pd
import numpy as np
import warnings
import random
import json
import cv2
from glob import glob
from typing import Any, Optional, List
import rasterio
from rasterio import logging
import csv
import xarray as xr
import torch.nn.functional as F
import torch.nn.functional as Fnn
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def _read_csv(infile):
    
    file = open(infile, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    return data[1:]

###########################################################################################################
# PACE Dataset
#############################################################################################################
class PaceDataset(Dataset):
    '''
    loads images from a csv filelist
    '''
    def __init__(self, input_dir=None,target_dir=None,csv_path =None, mode=None, transform=None):
        super().__init__()
        self.input_dir    = input_dir
        self.target_dir   = target_dir
        self.filelist     = _read_csv(csv_path)
        self.mode         = mode
        self.tensor_tranform = transforms.Compose([transforms.ToTensor()])
        if csv_path.endswith('.csv'):
            self.image_path = _read_csv(csv_path)
            self.indexing = True
        else:
            self.image_path = [fname for fname in os.listdir(csv_path) if fname.lower().endswith(('.npy'))]
            self.indexing = False
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.indexing:
            img_name     = str(self.image_path[idx][0])
        else:
            img_name     = str(self.image_path[idx])

        input_img  = np.nan_to_num(np.load(os.path.join(self.input_dir,self.filelist[idx][0])))
        output_img = None

        if self.mode=='cot':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][1])))
            output_img = np.log(output_img[:,:,0]+1)
        elif self.mode=='cer':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][1])))
            output_img = (output_img[:,:,1])/30.0
        elif self.mode=='cwp':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][1])))
            output_img = np.log(output_img[:,:,2]+1)
        elif self.mode=='cth':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][1])))
            output_img = (output_img[:,:,3])/10.0
        elif self.mode=='cmask':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][2])))
        elif self.mode=='all':
            output_img = np.nan_to_num(np.load(os.path.join(self.target_dir,self.filelist[idx][1])))
            output_img_cot = np.log(output_img[:,:,0]+1)
            output_img_cer = (output_img[:,:,1])/30.0
            output_img_cwp = np.log(output_img[:,:,2]+1)
            output_img_cth = (output_img[:,:,3])/10.0
            output_img = np.stack((output_img_cot,output_img_cer,output_img_cwp,output_img_cth),axis=2)
        
        
        # Convert to tensor         
        output_img =  self.tensor_tranform(output_img)

        if self.transform:
            input_img =  self.transform(input_img)  
        else:     
            input_img =  self.tensor_tranform(input_img)       
        
        return {'img':input_img, 'label':output_img}

    @staticmethod
    def build_transform(mean, std):
        """
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        t = []
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)



###########################################################################################################
def build_dataset(is_train: bool, args) -> Dataset:
    """
    Initializes a Dataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: Dataset object.
    """
    file_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'pace':
        mean = np.load('data_stats/mean.npy')
        std = np.load('data_stats/std.npy')        
        transform = PaceDataset.build_transform(mean, std) 
        input_dir = os.path.join(args.data_root_dir,'hsi')
        target_dir = os.path.join(args.data_root_dir,'target')
        dataset = PaceDataset(input_dir=input_dir,target_dir=target_dir,
                            csv_path = file_path, mode=args.data_mode, 
                            transform=transform)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset

def build_val_dataset(is_train: bool, args) -> Dataset:
    file_path = os.path.join(args.val_path)
    if args.dataset_type == 'pace':
        mean = np.load('data_stats/mean.npy')
        std = np.load('data_stats/std.npy')        
        transform = PaceDataset.build_transform(mean, std) 
        input_dir = os.path.join(args.data_root_dir,'hsi')
        target_dir = os.path.join(args.data_root_dir,'target')
        dataset = PaceDataset(input_dir=input_dir,target_dir=target_dir,
                            csv_path = file_path, mode=args.data_mode, 
                            transform=transform)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
