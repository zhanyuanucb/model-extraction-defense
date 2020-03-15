#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
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
import attack.utils.model as model_utils
import attack.utils.utils as attack_utils
import modelzoo.zoo as zoo
#from attack.victim.blackbox import Blackbox
#from attack.adversary.adv import RandomAdversary
import attack.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class PositiveNegativeSet:
    """ Dataset for loading positive samples"""

    def __init__(self, samples, normal_transform=None, random_transform=None, target_transform=None):
        assert normal_transform is not None, "PositiveSet: require vanilla normalization!"
        assert random_transform is not None, "PositiveSet: require random transformation!"
        self.samples = samples
        self.n_samples = len(self.samples)
        #self.targets = [s[1] for s in samples]
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, _ = self.samples[index]
        original = self.normal_transform(sample)
        random = self.random_transform(sample)
        # randomly choose a different image
        other_idx = random.choice(list(range(index) + list(index+1, self.n_samples)))
        other_sample, _ = self.samples[other_idx]
        other = self.normal_transform(other_sample)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        return original, random, other


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    print(device)
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of adversary\'s dataset (P_A(X))', default='CIFAR10')
    #parser.add_argument('--dataset_dir', metavar='TYPE', type=str, help='Directory of adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--epochs', metavar='TYPE', type=int, help='Training epochs', default=50)
    parser.add_argument('--optimizer_name', metavar='TYPE', type=str, help='Optimizer name', default="adam")

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    attack_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
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

    # Train/validation splitting
    sim_trainset = PositiveNegativeSet(trainset, normal_transform=transform, random_transform=random_transform)
    sim_valset = PositiveNegativeSet(valset, normal_transform=transform, random_transform=random_transform)
    #feat_trainset = TransferSetImagePaths(train_samples, transform=transform)
    #feat_valset = TransferSetImagePaths(val_samples, transform=transform)
    #sim_train_loader = DataLoader(sim_trainset, batch_size=1, shuffle=True, num_worker=10)

    model_name = params['model_name']
    num_classes = params['num_classes']
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    if gpu_count > 1:
       model = nn.DataParallel(model)
    model = model.to(device)

    epochs = params['epochs']
    optimizer_name = params["optimizer_name"]
    optimizer = get_optimizer(model.parameters(), optimizer_name)


    # Feature extraction training
    #model_utils.train_model(model, trainset, out_path, epochs=epochs, testset=valset,
                            checkpoint_suffix=".feat", device=device, optimizer=optimizer)
    
    # Or load a pretrained feature extractor

    # TODO: Implement similarity loss
    #sim_loss = model_utils.sim_loss
    #model_utils.train_model(model, sim_trainset, out_path, epochs=epochs, testset=sim_valset,
    #                        criterion_train=sim_loss, criterion_test=sim_loss,
    #                        checkpoint_suffix=".sim", device=device, optimizer=optimizer)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_root, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()