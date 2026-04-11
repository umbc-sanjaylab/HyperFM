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
    def __init__(self,
                 root_dir: str,
                 csv_path: str,
                 transform: None,
    ):
        """
        Creates dataset for multi-spectral single image classification.
        :param root_dir: path to all pace data files.
        :param transform: pytorch Transform for transforms and tensor conversion
        """
        super().__init__()
        self.root_dir   = root_dir
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
        """
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = torch.tensor(-1)  
        if self.indexing:
            img_name     = str(self.image_path[idx][0])
        else:
            img_name     = str(self.image_path[idx])

        abs_img_path = os.path.join(self.root_dir, img_name)
        images       = np.load(abs_img_path)

        images = np.nan_to_num(images)

        images = self.transform(images)   

        return {'img':images, 'label':labels}

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
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
        transform = PaceDataset.build_transform(is_train, args.input_size, mean, std) 
        data_dir = os.path.join(args.data_root_dir,'hsi')
        dataset = PaceDataset(data_dir,file_path, transform)  

    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset

def build_val_dataset(is_train: bool, args) -> Dataset:
    file_path = os.path.join(args.val_path)
    if args.dataset_type == 'pace':
        mean = np.load('data_stats/mean.npy')
        std = np.load('data_stats/std.npy')        
        transform = PaceDataset.build_transform(is_train, args.input_size, mean, std) 
        data_dir = os.path.join(args.data_root_dir,'hsi')
        dataset = PaceDataset(data_dir,file_path, transform)  
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
