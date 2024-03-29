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
import sklearn.metrics.pairwise as pairwise
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as tvdatasets
from torchvision.transforms import transforms as tvtransforms
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

from attack import datasets
import attack.utils.transforms as transform_utils
from attack.utils.transforms import *
import defense.similarity_encoding.encoder as encoder_utils
import attack.utils.model as model_utils
import attack.utils.utils as attack_utils
import modelzoo.zoo as zoo
import attack.config as cfg
from defense.utils import IdLayer

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"




from . import networks_basic as networks
self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    print(device)
gpu_count = torch.cuda.device_count()

# Reference: https://github.com/schoyc/blackbox-detection/blob/master/detection.py#L96
def calculate_thresholds(training_data, K, encoder=lambda x: x, P=1000, up_to_K=False):
    with torch.no_grad():
        data = encoder(training_data).numpy()
        #data = encoder(training_data)
    distances = []
    for i in range(data.shape[0]//P):
        distance_mat = pairwise.pairwise_distances(data[i * P:(i+1) * P,:], Y=data)
        distance_mat = np.sort(distance_mat, axis=-1)

        #distance_mat = torch.cdist(data[i * P:(i+1) * P,:], data, p=2)
        #distance_mat, _ = torch.sort(distance_mat, dim=-1)
        #distance_mat = distance_mat.numpy()

        distance_mat_K = distance_mat[:,1:K+1]
        distances.append(distance_mat_K)

    distance_matrix = np.concatenate(distances, axis=0)

    start = 1 if up_to_K else K

    THRESHORDS = []
    K_S = []
    for k in range(start, K+1):
        dist_to_k_neighbors = distance_matrix[:, :k]
        avg_dist_to_k_neighbors = dist_to_k_neighbors.mean(axis=-1)
        threshold = np.percentile(avg_dist_to_k_neighbors, 0.1)
        K_S.append(k)
        THRESHORDS.append(threshold)
    return K_S, THRESHORDS

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of adversary\'s dataset (P_A(X))', default='MNIST')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--model_suffix', metavar='TYPE', type=str, help='Model name', default="")
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument("--margins", nargs='+', type=float, required=True)
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--out_dir', metavar='TYPE', type=str, help='Save output to where', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--K', metavar='TYPE', type=int, help="K nearest neighbors", default=1000)
    parser.add_argument('--up_to_K', action="store_true")
    parser.add_argument('--norm', action="store_true")

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    #torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset_name']
    norm = params["norm"]
    dataset = datasets.__dict__[dataset_name]
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    if norm:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    else:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test2']

    # ---------------- Load dataset
    num_workers = params['nworkers']
    trainset = dataset(train=True, transform=transform)
    #train_dir = osp.join(cfg.DATASET_ROOT, modelfamily, 'MNIST/processed/train')
    #train_folder = ImageFolder(train_dir)
    #train_pathset = get_pathset(train_folder)
    #trainset = ImageDataGenerator(train_pathset, transform=transform)
    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=num_workers)

    # ---------------- Calculate thresholds for each encoder
    for margin in params["margins"]:
        #TODO: load LPIPS model

        K = params['K']
        up_to_K = params["up_to_K"]
        ks, thresholds = calculate_thresholds(train_data, K=K, encoder=model, up_to_K=up_to_K)

        out_dir = params['out_dir']
        out_dir = osp.join(out_dir, model_name, encoder_name)

        plt.plot(ks, thresholds, label=encoder_name)
        plt.xlabel('k (# of nearest neighbors)')
        plt.ylabel('Threshold (encodered space)')
        plt.title(f'Threshold vs k ({encoder_name})')
        plt.savefig(osp.join(out_dir, 'k_thresh_plot.png'), bbox_inches='tight')
        plt.clf()
        print(f"Save plot to {osp.join(out_dir, 'k_thresh_plot.png')}")

        with open(osp.join(out_dir, 'k_n_thresh.pkl'), 'wb') as file:
            print(f"Results saved to {osp.join(out_dir, 'k_n_thresh.pkl')}")
            pickle.dump([ks, thresholds], file)


if __name__ == '__main__':
    main()