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

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as tvdatasets

from attack import datasets
import attack.utils.transforms as transform_utils
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

def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--seedset_name', metavar='TYPE', type=str, help='Name of adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--seedset_dir', metavar='TYPE', type=str, help='Directory of adversary\'s dataset (P_A(X))', required=True)
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

    # ----------- Set up seedset
    seedset_name = params['seedset_name']
    seedset_dir = params['seedset_dir']
    valid_datasets = datasets.__dict__.keys()
    if seedset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[seedset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test2'] # Note: test2 has no normalization
    seedset = tvdatasets.ImageFolder(seedset_dir, transform=transform)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    seed_out_path = osp.join(out_path, 'seed.pt')
    adversary = RandomAdversary(blackbox, seedset, batch_size=batch_size)

    print('=> constructing seedset...')
    seedset = adversary.get_seedset()

    # ----------- Clean up transfer (top-1 predicted label)
    #new_transferset = []
    print('=> Using argmax labels (instead of posterior probabilities)')
    #for i in range(len(transferset)):
    #    x_i, y_i = transferset[i]
    #    argmax_k = y_i.argmax()
    #    y_i_1hot = torch.zeros_like(y_i)
    #    y_i_1hot[argmax_k] = 1.
    #    new_transferset.append((x_i, y_i_1hot))
    #transferset = new_transferset

    torch.save(seedset, seed_out_path)
    print('=> transfer set ({} samples) written to: {}'.format(seedset[0].size(0), seed_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_seed.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
