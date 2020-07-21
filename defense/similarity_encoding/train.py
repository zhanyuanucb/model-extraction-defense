#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import pickle
from PIL import Image
import json
from datetime import datetime
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
sys.path.append('/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding')
import numpy as np
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.datasets as tvdatasets
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torchvision.transforms import transforms as tvtransforms

from attack import datasets
import attack.utils.transforms as transform_utils
from attack.utils.transforms import *
import defense.similarity_encoding.encoder as encoder_utils
import attack.utils.model as model_utils
from attack.utils.model import get_optimizer
import attack.utils.utils as attack_utils
import modelzoo.zoo as zoo
import attack.config as cfg
from defense.utils import PositiveNegativeSet, IdLayer, BlinderPositiveNegativeSet
from attack.adversary.query_blinding.blinders import AutoencoderBlinders
import attack.adversary.query_blinding.transforms as blinders_transforms

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of dataset', default='CIFAR10')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--model_suffix', metavar='TYPE', type=str, default="")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('--train_epochs', metavar='TYPE', type=int, help='Training epochs', default=200)
    parser.add_argument('--sim_epochs', metavar='TYPE', type=int, help='Training epochs', default=50)
    parser.add_argument('--sim_norm', action='store_true')
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--callback', metavar='TYPE', type=float, help='Stop training once val acc meets requirement', default=None)
    parser.add_argument('--optimizer_name', metavar='TYPE', type=str, help='Optimizer name', default="adam")
    parser.add_argument('--ckpt_suffix', metavar='TYPE', type=str, default="")
    parser.add_argument('--margin', type=lambda x: np.sqrt(int(x)))
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--blinders_dir', metavar='PATH', type=str, default="/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_cinic10_0_")
    parser.add_argument('--resume', metavar='PATH', type=str, default=None)

    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker processes to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    attack_utils.create_dir(out_path)

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

    sim_norm = params["sim_norm"] # whether apply normalization on random_transform.
                                  # should be consistent to train/test_transform
    random_transform = transform_utils.RandomTransforms(modelfamily=modelfamily, normal=sim_norm)
    if sim_norm: # Apply data normalization
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    else:
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train2']
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test2']

    trainset = datasets.__dict__[dataset_name](train=True, transform=train_transform) # Augment data while training
    valset = datasets.__dict__[dataset_name](train=False, transform=test_transform)

    model_name = params['model_name']
    num_classes = params['num_classes']
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    model = model.to(device)

    train_epochs = params['train_epochs']
    optimizer_name = params["optimizer_name"]
    optimizer = get_optimizer(model.parameters(), optimizer_name)
    checkpoint_suffix = params["ckpt_suffix"]
    ckp_dir = params["ckp_dir"]
    load_pretrained = params['load_pretrained']
    callback = params['callback']
    if load_pretrained:
        ckp = osp.join(ckp_dir, f"checkpoint{checkpoint_suffix}.pth.tar")
        if not osp.isfile(ckp):
            print("=> no checkpoint found at '{}' but load_pretrained is {}".format(ckp, load_pretrained))
            exit(1)

    # ---------------- Feature extraction training
    if not load_pretrained:
        model_utils.train_model(model, trainset, out_path, epochs=train_epochs, testset=valset,
                                checkpoint_suffix=checkpoint_suffix, callback=callback, device=device, optimizer=optimizer)
    # ---------------- Or load a pretrained feature extractor
    else:
        print("=> loading checkpoint '{}'".format(ckp))
        checkpoint = torch.load(ckp)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Test accuracy: {best_test_acc}%")
    # -----------------------------------------------------
    
    # Build dataset for Positive/Negative samples
    train_dir = cfg.dataset2dir[dataset_name]["train"]
    test_dir = cfg.dataset2dir[dataset_name]["test"]

    # ----------------- Similarity training
    sim_trainset = PositiveNegativeSet(train_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)
    sim_valset = PositiveNegativeSet(test_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)

    #blinders_dir = params["blinders_dir"]
    #if blinders_dir is not None:
    #    blinders_ckp = osp.join(blinders_dir, "checkpoint.blind.pth.tar")
    #    if osp.isfile(blinders_ckp):
    #        blinders_noise_fn = blinders_transforms.get_gaussian_noise(device=device, r=0.095)
    #        auto_encoder = AutoencoderBlinders(blinders_noise_fn)
    #        print("=> Loading auto-encoder checkpoint '{}'".format(blinders_ckp))
    #        checkpoint = torch.load(blinders_ckp, map_location=device)
    #        start_epoch = checkpoint['epoch']
    #        best_test_loss = checkpoint['best_loss']
    #        auto_encoder.load_state_dict(checkpoint['state_dict'])
    #        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    #        print(f"===> Best val loss: {best_test_loss}")
    #        #auto_encoder = auto_encoder.to(device)
    #        auto_encoder.eval()
    #    else:
    #        print(f"Can't find auto-encoder ckp at {blinders_ckp}")
    #        exit(1)
    #sim_trainset = BlinderPositiveNegativeSet(train_dir, auto_encoder,
    #                                 normal_transform=test_transform,
    #                                 random_transform=random_transform)

    #sim_valset = BlinderPositiveNegativeSet(test_dir, auto_encoder,
    #                                 normal_transform=test_transform,
    #                                 random_transform=random_transform)

    # Replace the last layer
    #model.fc = IdLayer(activation=nn.Sigmoid()).to(device)
    activation_name = params['activation']
    if activation_name == "sigmoid":
        activation = nn.Sigmoid()
        print(f"Encoder activation: {activation_name}")
    else:
        print("Normal activation")
        activation = None
    model.fc = IdLayer(activation=activation).to(device)
    model = model.to(device)

    # Setup optimizer
    sim_optimizer = get_optimizer(model.parameters(), optimizer_type="sgdm", lr=1e-4, momentum=0.9)

    margin_train = params['margin']
    margin_test = margin_train
    sim_epochs = params['sim_epochs']
    adv_train = params['adv_train']
    checkpoint_suffix = ".sim-{:.1f}".format(margin_test)
    resume = params["resume"]

    model_suffix = params["model_suffix"]
    out_path = osp.join(out_path, model_name+model_suffix)
    if not osp.exists(out_path):
        os.mkdir(out_path)

    out_path = osp.join(out_path, "{}-margin-{:.1f}".format(dataset_name, margin_test))
    if not osp.exists(out_path):
        os.mkdir(out_path)

    encoder_utils.train_model(model, sim_trainset, out_path, epochs=sim_epochs, testset=sim_valset,
                            criterion_train=margin_train, criterion_test=margin_test,
                            checkpoint_suffix=checkpoint_suffix, device=device, optimizer=sim_optimizer, adv_train=adv_train, resume=resume)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, f'params_train{checkpoint_suffix}.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()