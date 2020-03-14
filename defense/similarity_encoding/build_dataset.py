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
import transform_utils.RandomTransforms as RandomTransforms
import attack.utils.model as model_utils
import attack.utils.utils as knockoff_utils
from attack.victim.blackbox import Blackbox
from attack.adversary.adv import RandomAdversary
import attack.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"


class PositiveNegativeSet(ImageFolder):
    """ Dataset for loading positive samples"""

    def __init__(self, samples, normal_transform=None, random_transform=None, target_transform=None):
        assert normal_transform is not None, "PositiveSet: require vanilla normalization!"
        assert random_transform is not None, "PositiveSet: require random transformation!"
        self.loader = deafult_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.n_samples = len(self.samples)
        #self.targets = [s[1] for s in samples]
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        original = self.normal_transform(sample)
        random = self.random_transform(sample)
        # randomly choose a different image
        other_idx = random.choice(list(range(index) + list(index+1, self.n_samples)))
        other_path, _ = self.samples[other_idx]
        other_sample = self.loader(other_path)
        other = self.normal_transform(other_sample)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        return original, random, other


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--dataset_dir', metavar='TYPE', type=str, help='Directory of adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if seedset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = transform_utils.RandomTransforms(modelfamily=modelfamily)
    dataset = datasets.__dict__[dataset_name](train=True, transform=transform)

    print([dataset.samples[i][0] for i in range(10)])

    # ----------- Clean up transfer (top-1 predicted label)
    new_transferset = []
    print('=> Using argmax labels (instead of posterior probabilities)')
    for i in range(len(transferset)):
        x_i, y_i = transferset[i]
        argmax_k = y_i.argmax()
        y_i_1hot = torch.zeros_like(y_i)
        y_i_1hot[argmax_k] = 1.
        new_transferset.append((x_i, y_i_1hot))
    transferset = new_transferset

    with open(seed_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), seed_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()


