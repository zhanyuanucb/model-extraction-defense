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
import matplotlib.pyplot as plt
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

from blinders import AutoencoderBlinders
import transforms as mytransforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Validate model')
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of validation dataset')
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('--num_demo', metavar='TYPE', type=int, default=5)
    parser.add_argument('--num_workers', metavar='TYPE', type=int, help='Number of processes of dataloader', default=10)

    args = parser.parse_args()
    params = vars(args)

    ckp_dir = params['ckp_dir']
    model_path = osp.join(ckp_dir, "checkpoint.blind.pth.tar")
    blinders_fn = mytransforms.get_gaussian_noise(device=device, sigma=0.095)
    auto_encoder = AutoencoderBlinders(blinders_fn)
    if osp.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_test_loss = checkpoint['best_loss']
        auto_encoder.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Best val loss: {best_test_loss}")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        exit(1)

    auto_encoder = auto_encoder.to(device)

    dataset_name = params["dataset_name"]
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transforms.ToTensor())
    num_workers = params['num_workers']
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=num_workers)

    for i, (image, label) in enumerate(test_loader, 1):
        image = image.to(device)
        with torch.no_grad():
            image_t = auto_encoder(image)
        fig = plt.figure(figsize=(8, 4))
        image = image[0].cpu().numpy().transpose([1, 2, 0])
        image_t = image_t[0].cpu().numpy().transpose([1, 2, 0])

        ax = plt.subplot(121)
        ax.set_title("original")
        plt.imshow(image)
        plt.axis("off")

        ax = plt.subplot(122)
        ax.set_title("original w/ blinders")
        plt.imshow(image_t)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(osp.join(ckp_dir, f"./demo_{i}.png"))
        if i == params["num_demo"]:
            break

if __name__ == '__main__':
    main()