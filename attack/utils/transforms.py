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

    def __init__(self, modelfamily="cifar", normal=False,
                 rotate_r=15, translate_r=0.05, scale_r=0.17, crop_r=0.04,
                 bright_r=0.09, contrast_r=0.55, unif_r=0.064, norm_std=0.095):
#    def __init__(self, modelfamily="cifar", normal=False,
#                 rotate_r=45, translate_r=0.45, scale_r=0.40, crop_r=0.3,
#                 bright_r=0.5, contrast_r=0.55, unif_r=0.1, norm_std=0.1):

        self.normal = normal
        if modelfamily == "cifar":
            self.normalize = transforms.Normalize(mean=cfg.CIFAR_MEAN,
                                             std=cfg.CIFAR_STD)
            self.size = 32
        elif modelfamily == "imagenet":
            self.normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN,
                                             std=cfg.IMAGENET_STD)
            self.size = 224
        elif modelfamily == "mnist":
            self.normalize = transforms.Normalize(mean=cfg.MNIST_MEAN,
                                             std=cfg.MNIST_STD)
            self.size = 28
        elif modelfamily == "cinic10":
            self.normalize = transforms.Normalize(mean=cfg.CINIC_MEAN,
                                             std=cfg.CINIC_STD)
            self.size = 32
        else:
            raise ValueError
        self.rotate_r = rotate_r
        self.translate_r = translate_r
        self.scale_r = scale_r
        self.crop_r = crop_r
        self.bright_r = bright_r
        self.contrast_r = contrast_r
        self.unif_r = unif_r 
        self.norm_std = norm_std

        self.candidates = [transforms.RandomRotation(self.rotate_r), # Rotation r=0.018
                           transforms.RandomAffine(0, translate=(self.translate_r, self.translate_r), resample=PIL.Image.BILINEAR), # Translate, r=0.45
                           transforms.RandomAffine(0, scale=(1-self.scale_r, 1+self.scale_r)), # Pixel-wise Scale, r=0.17
                           transforms.RandomResizedCrop(self.size, scale=(1-self.crop_r, 1.0), ratio=(1, 1)), # Crop and Resize, r=0.04, also resize from 3/4 to 4/3 (by default)
                           transforms.ColorJitter(brightness=self.bright_r), # Brightness, r=0.09
                           transforms.ColorJitter(contrast=self.contrast_r) # Contrast, r=0.55
                           ] if modelfamily != "mnist" else [
                           transforms.RandomRotation(self.rotate_r, fill=(0,)), # Rotation r=0.018
                           transforms.RandomAffine(0, translate=(self.translate_r, self.translate_r), resample=PIL.Image.BILINEAR, fillcolor=(0,)), # Translate, r=0.45
                           transforms.RandomAffine(0, scale=(1-self.scale_r, 1+self.scale_r), fillcolor=(0,)), # Pixel-wise Scale, r=0.17
                           transforms.RandomResizedCrop(self.size, scale=(1-self.crop_r, 1.0), ratio=(1, 1)), # Crop and Resize, r=0.04, also resize from 3/4 to 4/3 (by default)
                           transforms.ColorJitter(brightness=self.bright_r), # Brightness, r=0.09
                           transforms.ColorJitter(contrast=self.contrast_r) # Contrast, r=0.55
                           ]

        self.noise_candidates = [transforms.Lambda(lambda x: x + torch.randn_like(x).to(x.device) * self.norm_std),
                                              transforms.Lambda(lambda x: x + 2*self.unif_r*torch.rand_like(x).to(x.device) - self.unif_r)]
                                            
        self.noise = transforms.RandomChoice(self.noise_candidates)
        num_affine = len(self.candidates)
        num_noise = len(self.noise_candidates)
        self.noise_weight = num_noise/(num_noise + num_affine) 
        self.affine_weight = 1 - self.noise_weight
                
        self.noise = transforms.RandomChoice([transforms.Lambda(lambda x: x + torch.randn_like(x).to(x.device) * self.norm_std),
                                              transforms.Lambda(lambda x: x + 2*self.unif_r*torch.rand_like(x).to(x.device) - self.unif_r)]) 
        if self.normal:
            self.noise_transform = transforms.Compose([transforms.ToTensor(),
                                                       self.normalize,
                                                       self.noise])
            self.affinecolor_transform = transforms.Compose([transforms.RandomChoice(self.candidates),
                                                             transforms.ToTensor(),
                                                             self.normalize
                                                            ])
        else:
            self.noise_transform = transforms.Compose([transforms.ToTensor(),
                                                       self.noise])
            self.affinecolor_transform = transforms.Compose([transforms.RandomChoice(self.candidates),
                                                             transforms.ToTensor()
                                                            ])

    def __call__(self, x):
        t = random.choice([self.noise_transform, self.affinecolor_transform], p=[self.noise_weight, self.affine_weight])
        return t(x)