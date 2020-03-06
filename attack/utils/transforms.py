#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
from torchvision.transforms import transforms
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import attack.config as cfg

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
