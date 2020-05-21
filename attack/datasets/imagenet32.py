#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os
import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import attack.config as cfg

__author__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"


class ImageNet32(Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):
        if train:
            self.root = osp.join(cfg.DATASET_ROOT, 'ILSVRC2012_32x32', "out_lanczos_train")
        else:
            self.root = osp.join(cfg.DATASET_ROOT, 'ILSVRC2012_32x32', "out_lanczos_val")

        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []
        for data_batch in os.listdir(self.root):
            file_path = osp.join(self.root, data_batch)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.data)