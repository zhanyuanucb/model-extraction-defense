#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import attack.config as cfg
import attack.utils.utils as attack_utils

__original_author__ = "Tribhuvanesh Orekondy"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"



class AutoencoderBlinders(nn.Module):
    def __init__(self, blinders):
        super(AutoencoderBlinders, self).__init__()
        self.blinders = blinders
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 3, 3, padding=1)
        )
    
    def forward(self, x):
        noise = self.blinders(x)
        x_input = torch.cat([x, noise], dim=1)
        blinders = self.encoder(x_input)
        x_t = torch.clamp(x + blinders, 0., 1.)
        return x_t

# Reference: https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_step(model, train_loader, criterion, optimizer, epoch, device, scheduler, log_interval=10):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss, H, C = criterion(inputs)
        #if loss.item() == float("-inf"):
        #    print(H.item(), C.item())
        #    exit(1)
        loss.backward()
        optimizer.step()
        #scheduler.step(epoch)

        train_loss += loss.item()
        total += targets.size(0)

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                train_loss_batch))

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    return train_loss_batch


def test_step(model, test_loader, criterion, device, epoch=0., silent=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #print(f"testing batch {batch_idx}")
            inputs, targets = inputs.to(device), targets.to(device)
            loss, H, C = criterion(inputs)

            test_loss += loss.item()
            total += targets.size(0)

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\t'.format(epoch, test_loss))

    return test_loss


def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                callback=None,
                **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        attack_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    #if scheduler is None:
    #    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_loss = float("inf")
    best_test_loss = float("inf")

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> Resuming...")
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    #if osp.exists(log_path): # Remove previous log
    #    os.remove(log_path)
    #with open(log_path, 'w') as wf:
    #    columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
    #    wf.write('\t'.join(columns) + '\n')
    
    if not osp.exists(log_path): # Remove previous log
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'best_loss']
            wf.write('\t'.join(columns) + '\n')
    
    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        #scheduler.step(epoch) # should call optimizer.step() before scheduler.stop(epoch)
        train_loss = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           scheduler, log_interval=log_interval)
        best_train_loss = min(best_train_loss, train_loss)

        if test_loader is not None:
            #print("Start testing")
            test_loss = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_loss = min(best_test_loss, test_loss)

        # Checkpoint
        if test_loss <= best_test_loss:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_loss': test_loss,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, best_train_loss]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, best_test_loss]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
        
        # Callback
        if callback and test_acc >= callback:
            with open(log_path, 'a') as af:
                af.write(f'Validation accuracy reaches {callback}, so stop training.\n')
            return model, train_loader

    return model, train_loader