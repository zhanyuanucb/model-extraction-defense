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

import foolbox
from foolbox.attacks import LinfPGD as PGD
from foolbox.attacks import L2CarliniWagnerAttack
from foolbox.criteria import Misclassification, TargetedMisclassification

__original_author__ = "Tribhuvanesh Orekondy"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

def pgd_linf(model, x, y, eps=0.1, alpha=0.01, num_iter=20, criterion=nn.CrossEntropyLoss(), rand_init=False):
    # Reference: https://adversarial-ml-tutorial.org/adversarial_training/
    if rand_init:
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data*2*eps - eps
    else:
        delta = torch.zeros_like(x, requires_grad=True)

    for t in range(num_iter):
        loss = criterion(model(x+delta), y) #TODO
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return delta.detach()

def lossless_triplet(o_feat, t_feat, d_feat, dist=nn.MSELoss(), eps=1e-8):
    N = o_feat.size(-1)
    return -torch.log(-dist(t_feat, o_feat)/N + 1 + eps) - torch.log(-(N-dist(d_feat, o_feat))/N + 1 + eps)

def margin_loss(dist_fn, margin_sqr, anchor, positive, negative):
    return dist_fn(anchor, positive) + F.relu(margin_sqr - dist_fn(anchor, negative))

def get_triplet_loss(margin):
    return torch.nn.TripletMarginLoss(margin=margin)

def normalize(images, MEAN, STD):
    mean, std = MEAN.to(images.device), STD.to(images.device)
    return (images-mean) / std

def denormalize(images, MEAN, STD):
    mean, std = MEAN.to(images.device), STD.to(images.device)
    return torch.clamp(images*std + mean, 0., 1.)   

def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train_step(model, train_loader, margin, optimizer, epoch, device, scheduler, loss_fn=nn.MSELoss(), log_interval=10, adv_train=False):
    model.train()
    margin_sqr = margin**2
    train_loss = 0.
    p_correct = 0
    n_correct = 0
    total = 0
    epoch_size = len(train_loader.dataset)

    t_start = time.time()

    for batch_idx, (o, t, d, labels) in enumerate(train_loader):
        o, t, d, labels = o.to(device), t.to(device), d.to(device), labels.to(device)
        optimizer.zero_grad()
        o_feat = model(o)
        t_feat = model(t)
        d_feat = model(d)

        loss = loss_fn(t_feat, o_feat) + F.relu(margin_sqr - loss_fn(d_feat, o_feat))
        #loss = torch.norm(t_feat-o_feat, p=2, dim=1)

        #loss = loss_fn(o_feat, t_feat, d_feat)

        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        train_loss += loss.item()
        total += d.size(0)

        with torch.no_grad():
            p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
            n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)
        p_correct += p_dist.le(margin).sum().item()
        n_correct += n_dist.gt(margin).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        p_acc = 100. * p_correct / total
        n_acc = 100. * n_correct / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPositive Accuracy: {:.1f} ({}/{}) \tNegative Accuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(d), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), p_acc, p_correct, total, n_acc, n_correct, total))

    train_loss /= (batch_idx+1)
    t_end = time.time()
    t_epoch = int(t_end - t_start)

    return train_loss, p_acc, n_acc


def test_step(model, test_loader, margin, device, epoch=0., loss_fn=nn.MSELoss(), silent=False):
    model.eval()
    test_loss = 0.
    p_correct = 0
    n_correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (o, t, d, y) in enumerate(test_loader):
            o, t, d = o.to(device), t.to(device), d.to(device)
            o_feat = model(o)
            t_feat = model(t)
            d_feat = model(d)

            p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
            n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)

            loss = loss_fn(t_feat, o_feat) + F.relu(margin**2 - loss_fn(d_feat, o_feat))
            test_loss += loss.item()
            total += d.size(0)
            p_correct += p_dist.le(margin).sum().item()
            n_correct += n_dist.gt(margin).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    test_loss = test_loss / (batch_idx+1)

    p_acc = 100. * p_correct / total
    n_acc = 100. * n_correct / total

    if not silent:
        print('[Test]  Epoch: {} \tLoss: {:.6f}\tPositive Acc: {:.1f}% ({}/{})\tNegative Acc: {:.1f}% ({}/{})'.format(epoch, test_loss, p_acc, p_correct, total, n_acc, n_correct, total))
    return test_loss, p_acc, n_acc


def train_model(model, trainset, out_path, batch_size=32, margin_train=np.sqrt(10), margin_test=np.sqrt(10), testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, checkpoint_suffix='', optimizer=None, scheduler=None, adv_train=False,
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
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if osp.exists(log_path): # Remove previous log
        os.remove(log_path)
    with open(log_path, 'w') as wf:
        columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
        wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        #scheduler.step(epoch) # should call optimizer.step() before scheduler.stop(epoch)
        #loss_fn = get_triplet_loss(margin_train)
        loss_fn = nn.MSELoss()
        train_loss, train_pacc, train_nacc = train_step(model, train_loader, margin_train, optimizer, epoch, device,
                                           scheduler, loss_fn=loss_fn, log_interval=log_interval, adv_train=adv_train)
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
        #if test_nacc >= best_test_nacc: # Compare the negative sample accuracy
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

    # Save the one got the most trained
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

    return model, train_loader

#################################
# PGD adv. examples training
#################################
def foolbox_train_step(model, victim, attack, fmodel, MEAN, STD, objective_mode,
                       train_loader, margin, optimizer, epoch, device, scheduler, eps_factor=1., loss_fn=nn.MSELoss(), log_interval=10):
    model.train()
    margin_sqr = margin**2
    train_loss = 0.
    p_correct = 0
    n_correct = 0
    correct = 0
    total = 0
    epoch_size = len(train_loader.dataset)

    t_start = time.time()

    eps = 8./255*eps_factor
    for batch_idx, (o, d, labels) in enumerate(train_loader):
        o, d, labels = o.to(device), d.to(device), labels.to(device)
        with torch.no_grad():
            o_logits = victim(o)
        o_pred = o_logits.argmax(dim=-1)
        adv_criterion = Misclassification(o_pred)
        temp = denormalize(o, MEAN, STD)
        _, t, is_adv = attack(fmodel, temp, criterion=adv_criterion, epsilons=eps)
        t = normalize(t, MEAN, STD)
        #print(f"avg l2-dist of positive pairs: {foolbox.distances.l2(t, o).mean()}")
        #print(f"avg l2-dist of negative pairs: {foolbox.distances.l2(d, o).mean()}")

        optimizer.zero_grad()
        o_feat = model(o)
        t_feat = model(t)

        #loss = loss_fn(t_feat, o_feat) + F.relu(margin_sqr - loss_fn(d_feat, o_feat))
        if objective_mode == "sim_encoder":
            d_feat = model(d)
            loss = margin_loss(loss_fn, margin_sqr, o_feat, t_feat, d_feat)
        elif objective_mode == "binary_ood":
            positive_labels = np.zeros((o_feat.size(0), ), dtype=torch.long, device=device, requires_grad=True)
            negative_labels = np.ones((o_feat.size(0), ), dtype=torch.long, device=device, requires_grad=True)
            positive_loss = loss_fn(o_feat, positive_labels)
            negative_loss = loss_fn(t_feat, negative_labels)
            loss = positive_loss + negative_loss
        elif objective_mode == "adaptive_misinfo":
            uniform_labels = torch.full(o_feat.shape, fill_value=1/10., device=device, requires_grad=True)
            positive_loss = loss_fn(o_feat, labels)
            uniform_loss = soft_cross_entropy(t_feat, uniform_labels)
            loss = positive_loss + uniform_loss
        else:
            raise ValueError("Need to choose exactly one objective mode")

        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        train_loss += loss.item()
        total += d.size(0)

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        return_stat = None
        with torch.no_grad():
            if objective_mode == "sim_encoder":
                p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
                n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)
                p_correct += p_dist.le(margin).sum().item()
                n_correct += n_dist.gt(margin).sum().item()

                p_acc = 100. * p_correct / total
                n_acc = 100. * n_correct / total
                return_stat = (p_acc, n_acc)

                if (batch_idx + 1) % log_interval == 0:
                    print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPositive Accuracy: {:.1f} ({}/{}) \tNegative Accuracy: {:.1f} ({}/{})'.format(
                        exact_epoch, batch_idx * len(d), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item(), p_acc, p_correct, total, n_acc, n_correct, total))

            elif objective_mode == "binary_ood":
                positive_pred = o_feat.argmax(1)
                negative_pred = t_feat.argmax(1)
                correct += positive_pred.eq(positive_labels).sum().item()
                correct += negative_pred.eq(negative_labels).sum().item()
                acc = 100. * correct / (total*2)
                return_stat = acc

                if (batch_idx + 1) % log_interval == 0:
                    print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                        exact_epoch, batch_idx * len(o), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item(), acc, correct, total*2))

            elif objective_mode == "adaptive_misinfo":
                positive_conf = F.softmax(o_feat, dim=1)
                negative_conf = F.softmax(t_feat, dim=1)

                avg_positive_conf = positive_conf.argmax(1).mean()
                avg_negative_conf = negative_conf.argmax(1).mean()
                return_stat = (avg_positive_conf, avg_negative_conf)

                if (batch_idx + 1) % log_interval == 0:
                    print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tavg_positve_conf: {}\tavg_negative_conf: {}'.format(
                        exact_epoch, batch_idx * len(o), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item(), acc, avg_positive_conf, avg_negative_conf))

            else:
                raise ValueError("Need to choose exactly one objective mode")

    train_loss /= (batch_idx+1)
    t_end = time.time()
    t_epoch = int(t_end - t_start)

    return train_loss, return_stat


def foolbox_test_step(model, victim, attack, fmodel, MEAN, STD, objective_mode,
                      test_loader, margin, device, epoch=0., eps_factor=1., loss_fn=nn.MSELoss(), silent=False):
    model.eval()
    margin_sqr = margin**2
    test_loss = 0.
    p_correct = 0
    n_correct = 0
    total = 0
    t_start = time.time()
    eps = 8./255*eps_factor

    for batch_idx, (o, d, labels) in enumerate(test_loader):
        o, d, labels = o.to(device), d.to(device), labels.to(device)
        total += labels.size(0)
        ########################
        # Adv example crafting
        ########################
        with torch.no_grad():
            o_logits = victim(o)
        o_pred = o_logits.argmax(dim=-1)
        adv_criterion = Misclassification(o_pred)
        temp = denormalize(o, MEAN, STD)
        _, t, is_adv = attack(fmodel, temp, criterion=adv_criterion, epsilons=eps)
        t = normalize(t, MEAN, STD)
        ########################

        with torch.no_grad():
            o_feat = model(o)
            t_feat = model(t)

            if objective_mode == "sim_encoder":
                d_feat = model(d)
                loss = margin_loss(loss_fn, margin_sqr, o_feat, t_feat, d_feat)
                test_loss += loss.item()

                p_dist = torch.norm(o_feat - t_feat, p=2, dim=1)
                n_dist = torch.norm(o_feat - d_feat, p=2, dim=1)
                p_correct += p_dist.le(margin).sum().item()
                n_correct += n_dist.gt(margin).sum().item()
                p_acc = 100. * p_correct / total
                n_acc = 100. * n_correct / total
                return_stat = (p_acc, n_acc)
                if not silent:
                    print('[Test]  Epoch: {} \tLoss: {:.6f}\tPositive Acc: {:.1f}% ({}/{})\tNegative Acc: {:.1f}% ({}/{})'.format(epoch, test_loss, p_acc, p_correct, total, n_acc, n_correct, total))

            elif objective_mode == "binary_ood":
                positive_labels = np.zeros((o_feat.size(0), ), dtype=torch.long, device=device, requires_grad=True)
                negative_labels = np.ones((o_feat.size(0), ), dtype=torch.long, device=device, requires_grad=True)
                positive_loss = loss_fn(o_feat, positive_labels)
                negative_loss = loss_fn(t_feat, negative_labels)
                loss = positive_loss + negative_loss
                test_loss += loss.item()

                positive_pred = o_feat.argmax(1)
                negative_pred = t_feat.argmax(1)
                correct += positive_pred.eq(positive_labels).sum().item()
                correct += negative_pred.eq(negative_labels).sum().item()
                acc = 100. * correct / (total*2)
                return_stat = acc
                if not silent:
                    print('[Test]  Epoch: {} \tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\t'.format(epoch, test_loss, p_acc, p_correct, total))
            elif objective_mode == "adaptive_misinfo":
                uniform_labels = torch.full(o_feat.shape, fill_value=1/10., device=device, requires_grad=True)
                positive_loss = loss_fn(o_feat, labels)
                uniform_loss = soft_cross_entropy(t_feat, uniform_labels)
                loss = positive_loss + uniform_loss
                test_loss += loss.item()

                positive_conf = F.softmax(o_feat, dim=1)
                negative_conf = F.softmax(t_feat, dim=1)
                avg_positive_conf = positive_conf.argmax(1).mean()
                avg_negative_conf = negative_conf.argmax(1).mean()
                return_stat = (avg_positive_conf, avg_negative_conf)
                if not silent:
                    print('[Test]  Epoch: {} \tLoss: {:.6f}\tAvg Positive Conf: {:.1f} \tAvg Negative Conf: {:.1f}'.format(epoch, test_loss, p_acc, n_acc))
            else:
                raise ValueError("Need to choose exactly one objective mode")

            #loss = loss_fn(t_feat, o_feat) + F.relu(margin**2 - loss_fn(d_feat, o_feat))
            #test_loss += loss.item()
            #total += d.size(0)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    test_loss = test_loss / (batch_idx+1)

    return test_loss, return_stat


def foolbox_train_model(model, victim, trainset, out_path, mean, std, objective_mode, batch_size=32, margin_train=np.sqrt(10), margin_test=np.sqrt(10), testset=None,
                        device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                        epochs=100, log_interval=100, checkpoint_suffix='', optimizer=None, scheduler=None, eps_factor=1.,
                        **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        attack_utils.create_dir(out_path)
    run_id = str(datetime.now())
    MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1])
    STD = torch.Tensor(std).reshape([1, 3, 1, 1])

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
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if osp.exists(log_path): # Remove previous log
        os.remove(log_path)
    with open(log_path, 'w') as wf:
        columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
        wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    attack = PGD()
    fmodel = foolbox.models.PyTorchModel(victim, bounds=(0, 1), preprocessing={"mean":MEAN.to(device), "std":STD.to(device)})
    loss_fn = nn.MSELoss()
    for epoch in range(start_epoch, epochs + 1):
        #scheduler.step(epoch) # should call optimizer.step() before scheduler.stop(epoch)
        #loss_fn = get_triplet_loss(margin_train)
        # model, victim, attack, fmodel, 
        train_loss, return_stat = foolbox_train_step(model, victim, attack, fmodel, MEAN, STD, objective_mode,
                                                                train_loader, margin_train, optimizer, epoch, device,
                                                                scheduler, loss_fn=loss_fn, log_interval=log_interval, eps_factor=eps_factor)
        if objective_mode == "sim_encoder" or objective_mode == "adaptive_misinfo":
            train_pacc, train_nacc = return_stat
            best_train_pacc = max(best_train_pacc, train_pacc)
            best_train_nacc = max(best_train_nacc, train_nacc)
        elif objective_mode == "binary_ood":
            best_train_pacc = max(best_train_pacc, return_stat)
        else:
            raise ValueError("Need to choose exactly one objective mode")

        if test_loader is not None:
            #print("Start testing")
            test_loss, return_stat = foolbox_test_step(model, victim, attack, fmodel, MEAN, STD, objective_mode,
                                                                test_loader, margin_test, device, epoch=epoch, eps_factor=eps_factor)
            if objective_mode == "sim_encoder" or objective_mode == "adaptive_misinfo":
                test_pacc, test_nacc = return_stat
                best_test_pacc = max(best_test_pacc, test_pacc)
                best_test_nacc = max(best_test_nacc, test_nacc)
            elif objective_mode == "binary_ood":
                best_test_pacc = max(best_test_pacc, test_pacc)
            else:
                raise ValueError("Need to choose exactly one objective mode")

        # Checkpoint
        #if test_acc >= best_test_acc:
        #if test_loss <= best_test_loss: # Compare the loss
        #if test_nacc >= best_test_nacc: # Compare the negative sample accuracy
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

    # Save the one got the most trained
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

    return model, train_loader
