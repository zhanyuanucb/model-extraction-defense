import argparse
import torchvision.models as models
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp
import pickle
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset, DataLoader

import modelzoo.zoo as zoo
import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Validate model')
    parser.add_argument('--ckp_path', metavar='PATH', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of validation dataset')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes')
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=128)
    parser.add_argument('--num_workers', metavar='TYPE', type=int, help='Number of processes of dataloader', default=10)

    args = parser.parse_args()
    params = vars(args)

    model_path = params['ckp_path']
    model_name = params['model_name']
    num_classes = params['num_classes']
    dataset_name = params['dataset_name']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    if osp.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        exit(1)

    if gpu_count > 1:
       model = nn.DataParallel(model)
    model = model.to(device)

    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=None)

    tic = time.time()
    model_utils.test_step(model, test_loader, criterion_test, device)
    tac = time.time()
    print("validation time: {} min".format((tac - tic)/60))

if __name__ == '__main__':
    main()