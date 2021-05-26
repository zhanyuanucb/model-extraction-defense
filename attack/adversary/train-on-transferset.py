#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
from datetime import datetime
import json
from collections import defaultdict as dd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

import attack.config as cfg
from attack import datasets
import attack.utils.transforms as transform_utils
import attack.utils.model as model_utils
import attack.utils.utils as knockoff_utils
import modelzoo.zoo as zoo

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    #parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
    #                    default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default="adam")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--num_classes', type=int, help='number of classes', default=10)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--transferlog_name', default=None, type=str, metavar='PATH',
                        help='path to transferset')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--log_suffix', type=str, help='log suffix', default=".no_transform")
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    #train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    try:
        #trainset = dataset(train=True, transform=train_transform)
        testset = dataset(train=False, transform=test_transform)
    except TypeError as e:
        #trainset = dataset(split="train", transform=train_transform)
        testset = dataset(split="valid", transform=test_transform)

    #num_classes = len(trainset.classes)
    num_classes = params['num_classes']

    ##################################
    # Load from one log
    ##################################
    #transferset_dir = f"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/{params['transferlog_name']}/transferset.pt"

    #trainset = torch.load(transferset_dir)
    #trainset.transform=test_transform

    ##################################
    # Load from several logs
    ##################################
    log_name = "2021-05-02_23-12-18-fbtop3-cwl2_{CIFAR10}_seed500"
    log_name_oracle = "2021-05-03_18-19-48-fbtop3-cwl2_{CIFAR10}_seed500_oracle"
    transferset_dir = f"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/{log_name}/transferset.pt"
    oracle_transferset_dir = f"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/{log_name_oracle}/transferset.pt"

    regular_trainset = torch.load(transferset_dir)
    regular_trainset.transform=test_transform
    oracle_trainset = torch.load(oracle_transferset_dir)
    oracle_trainset.transform=test_transform
    trainset = ConcatDataset([regular_trainset, oracle_trainset]) 

    subset_idxs = np.random.choice(range(len(trainset)), size=50000, replace=False)
    trainset = Subset(trainset, subset_idxs)


    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    modelfamily = 'cifar'
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    # ----------- Set up optimizer
    optim_type = params['optimizer']
    lr = params['lr']
    momentum = params['momentum']
    optimizer = model_utils.get_optimizer(model.parameters(), optim_type, lr=lr, momentum=momentum)

    # Store arguments
    params["out_path"] = f"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/{params['transferlog_name']}/"

    #out_path = osp.join(out_path, f"{params['dataset']}-{params['model_arch']}")
    #if not osp.exists(out_path):
    #    os.mkdir(out_path)

    params['created_on'] = str(datetime.now())
    #params_out_path = osp.join(out_path, 'params.json')
    #with open(params_out_path, 'w') as jf:
    #    json.dump(params, jf, indent=True)

    params['optimizer'] = optimizer
    #params['scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, 60)

    #params['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0.001)
    # ----------- Train
    model_utils.train_model(model, trainset, testset=testset, checkpoint_suffix=".no_transform",
                            device=device, criterion_train=model_utils.soft_cross_entropy, **params)



if __name__ == '__main__':
    main()
