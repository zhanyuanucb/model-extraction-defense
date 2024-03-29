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
from sklearn.model_selection import train_test_split
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
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
from attack.victim.blackbox import Blackbox
from defense.utils import PositiveNegativeSet, PositiveNegativeImageSet, IdLayer, BlinderPositiveNegativeSet
from defense.utils import LogisticLayer
from defense.utils import BinarySampleSet
from attack.adversary.query_blinding.blinders import AutoencoderBlinders
import attack.adversary.query_blinding.transforms as blinders_transforms

import foolbox
print(f"foolbox version: {foolbox.__version__}")
from foolbox.attacks import LinfPGD as PGD
from foolbox.attacks import L2CarliniWagnerAttack
from foolbox.criteria import Misclassification, TargetedMisclassification

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
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
    parser.add_argument('--train_on_seed', action='store_true')
    parser.add_argument('--sim_encoder', action='store_true')
    parser.add_argument('--foolbox_train', action='store_true')
    parser.add_argument('--adaptive_misinfo', action='store_true')
    parser.add_argument('--binary_ood', action='store_true')
    parser.add_argument('--seedsize', metavar='TYPE', type=int, help='size of seed images', default=5000)
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--eps_factor', metavar='TYPE', type=float, help='PGD eps factor', default=1.)
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

    feat_out_path = osp.join(params['out_dir'], "feat_extractor", f"{params['dataset_name']}-{params['model_name']}-{params['model_suffix']}")
    attack_utils.create_dir(feat_out_path)

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
        #train_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    else:
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train2']
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test2']
    try:
        trainset = datasets.__dict__[dataset_name](train=True, transform=train_transform) # Augment data while training
        valset = datasets.__dict__[dataset_name](train=False, transform=test_transform)
    except TypeError as e:
        trainset = datasets.__dict__[dataset_name](split="train", transform=train_transform) # Augment data while training
        valset = datasets.__dict__[dataset_name](split="valid", transform=test_transform)
    ##########################################
    # Using seed images
    ##########################################
    if params['train_on_seed']:
        trainset_full = trainset
        seed_idx = np.random.choice(range(len(trainset)), size=params['seedsize'], replace=False)
        train_idx, val_idx = train_test_split(seed_idx, test_size=0.1, random_state=42)
        trainset = Subset(trainset_full, train_idx)
        valset = Subset(trainset_full, val_idx)

    model_name = params['model_name']
    num_classes = params['num_classes']
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    model = model.to(device)

    train_epochs = params['train_epochs']
    optimizer_name = params["optimizer_name"]
    optimizer = get_optimizer(model.parameters(), optimizer_name, lr=1e-3)
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
        model_utils.train_model(model, trainset, feat_out_path, epochs=train_epochs, testset=valset,
                                checkpoint_suffix=checkpoint_suffix, callback=callback, device=device, optimizer=optimizer)
        #print("Train encoder w/ random feature extractor")
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

#        ####################
#        # Adversarial PNSet
#        ####################
#
#        MEAN, STD = cfg.NORMAL_PARAMS['cifar']
#        adv_criterion = Misclassification()
#        attack = PGD()
#        eps = 8./256
#        fmodel = foolbox.models.PyTorchModel(victim, bounds=(0, 1), preprocessing={"mean":MEAN.to(device), "std":STD.to(device)})
#
#        sim_trainset = AdvPositiveNegativeSet(attack, fmodel, eps, adv_criterion,
#                                              train_dir, MEAN, STD, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)
#        sim_valset = AdvPositiveNegativeSet(attack, fmodel, eps, adv_criterion,
#                                              test_dir, MEAN, STD, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)

    if params["foolbox_train"]:
        sim_trainset = BinarySampleSet(train_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)

        sim_valset = BinarySampleSet(test_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)
        #sim_valset = None
        blackbox_dir = "/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28_2"
        victim = Blackbox.from_modeldir(blackbox_dir, device, output_type="logits")
        victim = victim.model
    else:
        try:    
            # ----------------- Similarity training
            sim_trainset = PositiveNegativeSet(train_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)

            sim_valset = PositiveNegativeSet(test_dir, normal_transform=test_transform, random_transform=random_transform, dataset=dataset_name)
        except IsADirectoryError as e:
            # ----------------- Similarity training
            sim_trainset = PositiveNegativeImageSet(train_dir, normal_transform=test_transform, random_transform=random_transform)

            sim_valset = PositiveNegativeImageSet(test_dir, normal_transform=test_transform, random_transform=random_transform)

    ################################
    # Adversary only use seed images
    ################################
    if params["train_on_seed"]:
        print("=> Train on seedset")
        sim_trainset.data, sim_trainset.targets = np.transpose(trainset.dataset.data[trainset.indices], (0, 3, 1, 2)), [trainset.dataset.targets[idx] for idx in trainset.indices]
        sim_trainset.n_samples = sim_trainset.data.shape[0]

        sim_valset.data, sim_valset.targets = np.transpose(valset.dataset.data[valset.indices], (0, 3, 1, 2)), [valset.dataset.targets[idx] for idx in valset.indices]
        sim_valset.n_samples = sim_valset.data.shape[0]


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
    if params["sim_encoder"]:
        objective_mode = "sim_encoder"
        activation_name = params['activation']
        if activation_name == "sigmoid":
            activation = nn.Sigmoid()
            print(f"Encoder activation: {activation_name}")
        else:
            print("Normal activation")
            activation = None
        model.fc = IdLayer(activation=activation).to(device)
    elif params["binary_ood"]:
        objective_mode = "binary_ood"
        feature_dim = params["feature_dim"]
        model.fc = LogisticLayer(feature_dim)
    elif params["adaptive_misinfo"]:
        objective_mode = "adaptive_misinfo"
    else:
        raise ValueError("Need to choose exactly one objective mode")

    model = model.to(device)

    # Setup optimizer
    #sim_optimizer = get_optimizer(model.parameters(), optimizer_type="sgdm", lr=1e-4, momentum=0.9)
    sim_optimizer = get_optimizer(model.parameters(), optimizer_type="adam", lr=1e-3)

    margin_train = params['margin']
    margin_test = margin_train
    sim_epochs = params['sim_epochs']
    adv_train = params['adv_train']
    eps_factor = params['eps_factor']
    checkpoint_suffix = ".sim-{:.1f}".format(margin_test)
    resume = params["resume"]

    model_suffix = params["model_suffix"]
    sim_out_path = osp.join(params['out_dir'], model_name+model_suffix)
    attack_utils.create_dir(sim_out_path)

    sim_out_path = osp.join(sim_out_path, "{}-margin-{:.1f}".format(dataset_name, margin_test))
    attack_utils.create_dir(sim_out_path)

    if params["foolbox_train"]:
        MEAN, STD = cfg.NORMAL_PARAMS['cifar']
        encoder_utils.foolbox_train_model(model, victim, sim_trainset, sim_out_path, MEAN, STD, objective_mode,
                                epochs=sim_epochs, testset=sim_valset,
                                criterion_train=margin_train, criterion_test=margin_test,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=sim_optimizer, eps_factor=eps_factor, resume=resume)

    else:
        assert objective_mode=="sim_encoder", f"regular encoder training only support sim_encoder, but got {objective_mode}"
        encoder_utils.train_model(model, sim_trainset, sim_out_path, epochs=sim_epochs, testset=sim_valset,
                                criterion_train=margin_train, criterion_test=margin_test,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=sim_optimizer, adv_train=adv_train, resume=resume)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(sim_out_path, f'params_train{checkpoint_suffix}.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()