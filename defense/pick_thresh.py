#!/usr/bin/python
"""
Find threshold for detector
"""
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')

import numpy as np
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as tvdatasets
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

from attack import datasets
import attack.utils.transforms as transform_utils
from attack.utils.transforms import *
import defense.similarity_encoding.encoder as encoder_utils
import attack.utils.model as model_utils
import attack.utils.utils as attack_utils
import modelzoo.zoo as zoo
import attack.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

class IdLayer(nn.Module):
    def __init__(self):
        super(IdLayer, self).__init__()
    def forward(self, x):
        return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    print(device)
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of adversary\'s dataset (P_A(X))', default='CIFAR10')
    #parser.add_argument('--dataset_dir', metavar='TYPE', type=str, help='Directory of adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('--epochs', metavar='TYPE', type=int, help='Training epochs', default=50)
    parser.add_argument('--optimizer_name', metavar='TYPE', type=str, help='Optimizer name', default="adam")

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    ckp = params['ckp_dir']

    #torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    random_transform = transform_utils.RandomTransforms(modelfamily=modelfamily)
    trainset = datasets.__dict__[dataset_name](train=True, transform=transform)
    valset = datasets.__dict__[dataset_name](train=False, transform=transform)

    model_name = params['model_name']
    num_classes = params['num_classes']
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    #if gpu_count > 1:
    #   model = nn.DataParallel(model)
    model.last_linear = IdLayer().to(device)
    model = model.to(device)

    # Load a pretrained similarity encoder
    if osp.isfile(ckp):
        print("=> loading checkpoint '{}'".format(ckp))
        checkpoint = torch.load(ckp)
        best_pacc = checkpoint['best_pacc']
        best_nacc = checkpoint['best_nacc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint:\n best_pacc: {} \n best_nacc: {}".format(best_pacc, best_nacc))
    else:
        print("=> no checkpoint found at '{}'".format(ckp))
    
    # Replace the last layer
    model.last_linear = IdLayer().to(device)
    #if gpu_count > 1:
    #   model = nn.DataParallel(model)
    model = model.to(device)

    batch_size = params['batch_size']
    num_workers = params['num_workers']
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    encoded = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            encoded_images = model(images)
            encoded.extend([encoded_images[i] for i in range(encoded_images.size(0))])

if __name__ == '__main__':
    main()