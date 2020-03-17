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

def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train_step(model, train_loader, margin, optimizer, epoch, device, scheduler, log_interval=10):
    model.train()
    train_loss = 0.
    p_correct = 0
    n_correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)

    t_start = time.time()

    for batch_idx, (o, t, d) in enumerate(train_loader):
        """
        # ------------------- Start debugging session
        print(o.shape, t.shape, d.shape)
        mean = torch.tensor(cfg.CIFAR_MEAN).reshape((3, 1, 1))
        std = torch.tensor(cfg.CIFAR_STD).reshape((3, 1, 1))
        import matplotlib.pyplot as plt
        debug_root = osp.join('/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding', 'debug')
        if not osp.exists(debug_root):
            os.mkdir(debug_root)
        for i in range(20):
            save_path = osp.join(debug_root, str(i))
            if not osp.exists(save_path):
                os.mkdir(save_path)
            o_i, t_i, d_i = o[i], t[i], d[i]
            o_i = np.clip((o_i).numpy().transpose([1, 2, 0]), 0., 1.)
            t_i = np.clip((t_i).numpy().transpose([1, 2, 0]), 0., 1.)
            d_i = np.clip((d_i).numpy().transpose([1, 2, 0]), 0., 1.)
            ot_i = np.clip(np.abs(o_i - t_i), 0., 1.)
            plt.imsave(osp.join(save_path, 'o.png'), o_i)
            plt.imsave(osp.join(save_path, 't.png'), t_i)
            plt.imsave(osp.join(save_path, 'd.png'), d_i)
            plt.imsave(osp.join(save_path, 'ot.png'), ot_i)

        exit(1)
        # ------------------ End debugging session
        """
        o, t, d = o.to(device), t.to(device), d.to(device)
        optimizer.zero_grad()
        o_feat = model(o)
        t_feat = model(t)
        d_feat = model(d)
        p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
        n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)
        loss = torch.mean(p_dist**2 + F.relu(margin**2 - n_dist**2))
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        train_loss += loss.item()
        total += d.size(0)

        p_correct += p_dist.le(margin).sum().item()
        n_correct += n_dist.gt(margin).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        p_acc = 100. * p_correct / total
        n_acc = 100. * n_correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPositive Accuracy: {:.1f} ({}/{}) \tNegative Accuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(d), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), p_acc, p_correct, total, n_acc, n_correct, total))

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    return train_loss_batch, p_acc, n_acc


def test_step(model, test_loader, margin, device, epoch=0., silent=False):
    model.eval()
    test_loss = 0.
    p_correct = 0
    n_correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (o, t, d) in enumerate(test_loader):
            o, t, d = o.to(device), t.to(device), d.to(device)
            o_feat = model(o)
            t_feat = model(t)
            d_feat = model(d)
            p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
            n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)
            loss = torch.mean(p_dist**2 + F.relu(margin**2 - n_dist**2))
            test_loss += loss.item()
            total += d.size(0)
            p_correct += p_dist.le(margin).sum().item()
            n_correct += n_dist.gt(margin).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    test_loss = test_loss / total

    p_acc = 100. * p_correct / total
    n_acc = 100. * n_correct / total

    if not silent:
        print('[Test]  Epoch: {} \tLoss: {:.6f}\tPositive Acc: {:.1f}% ({}/{})\tNegative Acc: {:.1f}% ({}/{})'.format(epoch, test_loss, p_acc, p_correct, total, n_acc, n_correct, total))
    return test_loss, p_acc, n_acc


def train_model(model, trainset, out_path, batch_size=64, margin_train=np.sqrt(10), margin_test=np.sqrt(10), testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, checkpoint_suffix='', optimizer=None, scheduler=None,
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

    # Optimizer
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_pacc, best_train_nacc, train_pacc, train_nacc = -1., -1., -1., -1.
    best_test_pacc, best_test_nacc, test_pacc, test_nacc, best_test_loss, test_loss,  = -1., -1., -1., -1., -float('inf'), -float('inf')


    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'p_acc', 'best_pacc', 'n_acc', 'best_nacc']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        #scheduler.step(epoch) # should call optimizer.step() before scheduler.stop(epoch)
        train_loss, train_pacc, train_nacc = train_step(model, train_loader, margin_train, optimizer, epoch, device,
                                           scheduler, log_interval=log_interval)
        best_train_pacc = max(best_train_pacc, train_pacc)
        best_train_nacc = max(best_train_nacc, train_nacc)


        if test_loader is not None:
            #print("Start testing")
            test_loss, test_pacc, test_nacc = test_step(model, test_loader, margin_test, device, epoch=epoch)
            best_test_pacc = max(best_test_pacc, test_pacc)
            best_test_nacc = max(best_test_nacc, test_nacc)

        # Checkpoint
        #if test_acc >= best_test_acc:
        #if test_loss <= best_test_loss: # Compare the loss
        if test_nacc >= best_test_nacc: # Compare the negative sample accuracy
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_pacc': test_pacc,
                'best_nacc': test_nacc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_pacc, best_train_pacc, train_nacc, best_train_nacc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_pacc, best_test_pacc, test_nacc, best_test_nacc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model, train_loader
