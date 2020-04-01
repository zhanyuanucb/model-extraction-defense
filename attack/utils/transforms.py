#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
from torchvision.transforms import transforms
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import attack.config as cfg
from numpy import random
import PIL
import torch

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

class DefaultTransforms:
    normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN,
                                     std=cfg.IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

class RandomTransforms:

    def __init__(self, modelfamily="cifar"):
        if modelfamily == "cifar":
            #self.normalize = transforms.Normalize(mean=cfg.CIFAR_MEAN,
            #                                 std=cfg.CIFAR_STD)
            self.size = 32
        elif modelfamily == "imagenet":
            #self.normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN,
            #                                 std=cfg.IMAGENET_STD)
            self.size = 224
        else:
            raise ValueError

        self.candidates = [transforms.RandomRotation(0.018), # Rotation r=0.018
                           transforms.RandomAffine(0, translate=(0.45, 0.45), resample=PIL.Image.BILINEAR), # Translate, r=0.45
                           transforms.RandomAffine(0, scale=(1-0.17, 1+0.17)), # Pixel-wise Scale, r=0.17
                           transforms.RandomResizedCrop(self.size, scale=(1-0.04, 1.0), ratio=(1, 1)), # Crop and Resize, r=0.04, also resize from 3/4 to 4/3 (by default)
                           transforms.ColorJitter(brightness=0.09), # Brightness, r=0.09
                           transforms.ColorJitter(contrast=0.55) # Contrast, r=0.55
                           ]
                
        self.noise = transforms.RandomChoice([transforms.Lambda(lambda x: x + torch.randn_like(x).to(x.device) * 0.095),
                                              transforms.Lambda(lambda x: x + 0.128*torch.rand_like(x).to(x.device) - 0.064)]) 
        self.noise_transform = transforms.Compose([transforms.ToTensor(),
                                                   #self.normalize,
                                                   self.noise])
        self.affinecolor_transform = transforms.Compose([transforms.RandomChoice(self.candidates),
                                                         transforms.ToTensor(),
                                                         #self.normalize]
                                                        ])
        self.random_transform = transforms.RandomChoice([self.noise_transform, self.affinecolor_transform])


    def __call__(self, x):
        t = random.choice([self.noise_transform, self.affinecolor_transform], p=[0.25, 0.75])
        return t(x)