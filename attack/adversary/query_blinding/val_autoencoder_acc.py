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
from torchvision.transforms import transforms

import modelzoo.zoo as zoo
import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets

import blind as blind_utils
import transforms as mytransforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Validate model')
    parser.add_argument('--auto_path', metavar='PATH', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--ckp_path', metavar='PATH', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of validation dataset')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes')
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=128)
    parser.add_argument('--num_workers', metavar='TYPE', type=int, help='Number of processes of dataloader', default=10)

    args = parser.parse_args()
    params = vars(args)

    auto_path = params['auto_path']
    num_classes = params['num_classes']
    dataset_name = params['dataset_name']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    MEAN, STD = cfg.NORMAL_PARAMS[modelfamily]
    MEAN, STD = torch.Tensor(MEAN).reshape([1, 3, 1, 1]), torch.Tensor(STD).reshape([1, 3, 1, 1])
    MEAN, STD = MEAN.to(device), STD.to(device)

    # ---------------- Set up Auto-encoder
    blinders = mytransforms.get_gaussian_noise(device=device, sigma=0.095)
    auto_encoder = blind_utils.AutoencoderBlinders(blinders)
    auto_encoder = auto_encoder.to(device)

    auto_path = osp.join(auto_path, "checkpoint.blind.pth.tar")
    if osp.isfile(auto_path):
        print("=> Loading auto-encoder checkpoint '{}'".format(auto_path))
        checkpoint = torch.load(auto_path)
        start_epoch = checkpoint['epoch']
        best_test_loss = checkpoint['best_loss']
        auto_encoder.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Best val loss: {best_test_loss}%")
    else:
        print("=> no checkpoint found at '{}'".format(auto_path))
        exit(1)

    auto_encoder = auto_encoder.to(device)
    auto_encoder.eval()
    # -----------------

    # ---------------- Set up Classifier
    model_name = params['model_name']
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)
    ckp_path = params['ckp_path']
    ckp_path = osp.join(ckp_path, "checkpoint.pth.tar")
    if osp.isfile(ckp_path):
        print("=> Loading classifier checkpoint '{}'".format(ckp_path))
        checkpoint = torch.load(ckp_path)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Best val loss: {best_test_acc}%")
    else:
        print("=> no checkpoint found at '{}'".format(ckp_path))
        exit(1)

    model = model.to(device)
    model.eval()
    # -----------------

    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transforms.ToTensor())
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loss = 0.
    correct = 0
    total = 0
    tic = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            x_t = auto_encoder(inputs)
            x_norm = (x_t - MEAN) / STD
            outputs = model(x_norm)
            nclasses = outputs.size(1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    tac = time.time()

    acc = 100. * correct / total

    print(f"Test Acc.: {acc}%")
    print("validation time: {} min".format((tac - tic)/60))

if __name__ == '__main__':
    main()