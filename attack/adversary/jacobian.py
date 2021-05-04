#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
from defense.utils import ImageTensorSet, ImageTensorSetKornia
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import time
import re

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision

import foolbox
from foolbox.attacks import LinfPGD as PGD
from foolbox.attacks import L2CarliniWagnerAttack
from foolbox.criteria import Misclassification, TargetedMisclassification

from attack import datasets
import attack.utils.transforms as transform_utils
import attack.utils.model as model_utils
import attack.utils.utils as knockoff_utils
import attack.config as cfg
import modelzoo.zoo as zoo

from defense.detector import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def make_one_hot(labels, K):
    return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)


class JacobianAdversary:
    """
    PyTorch implementation of:
    1. (JBDA) "Practical Black-Box Attacks against Machine Learning", Papernot et al., ACCS '17
    2. (JB-{topk, self}) "PRADA: Protecting against DNN Model Stealing Attacks", Juuti et al., Euro S&P '19
    """
    def __init__(self, blackbox, budget, model_adv_name, model_adv_pretrained, modelfamily, seedset, testset, device, out_dir, aug_batch_size=128,
                 num_classes=10, batch_size=cfg.DEFAULT_BATCH_SIZE, ema_decay=-1, detector=None, blinder_fn=None, binary_search=False, use_feature_fool=False, foolbox_alg="pgd",
                 eps=0.1, num_steps=8, train_epochs=20, kappa=400, tau=None, rho=6, sigma=-1, take_lastk=-1,
                 query_batch_size=1, random_adv=False, adv_transform=False, aug_strategy='jbda', useprobs=True, final_train_epochs=100):
        self.blackbox = blackbox
        self.budget = budget
        self.model_adv_name = model_adv_name
        self.model_adv_pretrained = model_adv_pretrained
        self.model_adv = None
        self.modelfamily = modelfamily
        self.seedset = seedset
        self.testset = testset
        self.batch_size = batch_size
        self.aug_batch_size = aug_batch_size
        self.query_batch_size = query_batch_size
        self.ema_decay = ema_decay
        self.detector_adv = detector
        self.blinder_fn = blinder_fn
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, pin_memory=True)
        self.train_epochs = train_epochs
        self.final_train_epochs = final_train_epochs
        self.eps = eps
        self.num_steps = num_steps
        self.kappa = kappa
        self.tau = tau
        self.rho = rho
        self.sigma = sigma
        self.take_lastk = take_lastk
        self.binary_search = binary_search
        self.use_feature_fool = use_feature_fool
        self.foolbox_alg=foolbox_alg
        self.device = device
        self.MEAN, self.STD = cfg.NORMAL_PARAMS[modelfamily]
        self.MEAN, self.STD = torch.Tensor(self.MEAN), torch.Tensor(self.STD)
        self.adv_transform = adv_transform
        if modelfamily != "mnist":
            self.MEAN = self.MEAN.reshape([1, 3, 1, 1])
            self.STD = self.STD.reshape([1, 3, 1, 1])
        self.out_dir = out_dir
        self.num_classes = num_classes
        self.random_adv = random_adv
        assert (aug_strategy in ['jbda', 'jbself']) or 'jbtop' in aug_strategy or 'fbtop' in aug_strategy
        self.aug_strategy = aug_strategy
        self.topk = 0
        if 'jbtop' in aug_strategy:
            # extract k from "jbtop<k>"
            self.topk = int(aug_strategy.replace('jbtop', ''))
        if 'fbtop' in aug_strategy:
            self.topk = int(aug_strategy.replace('fbtop', ''))

        self.accuracies = []  # Track test accuracies over time
        self.useprobs = useprobs
        self.log_path = osp.join(self.out_dir, "advaug.log.tsv")

        self.images2inputs, self.imagesb2inputs, self.feat2feat_b, self.feat2feat_org = [], [], [], []

        # -------------------------- Initialize log
        if self.log_path:
            self._init_log()

        # -------------------------- Initialize seed data
        print('=> Obtaining predictions over {} seed samples using strategy {}'.format(len(self.seedset),
                                                                                       self.aug_strategy))
        Dx = torch.cat([self.seedset[i][0].unsqueeze(0) for i in range(len(self.seedset))])
        Dy = []

        # Populate Dy
        with torch.no_grad():
            for inputs, in DataLoader(TensorDataset(Dx), batch_size=self.query_batch_size, shuffle=False):
                inputs = inputs.to(self.device)
                try: # No detector
                    if isinstance(blackbox, ELBODetector2):
                        outputs = blackbox(inputs, None, is_init=True, is_adv=True).cpu()
                    else:
                        outputs = blackbox(inputs, is_adv=True).cpu()
                except TypeError as e:
                    outputs = blackbox(inputs).cpu()
                #if not self.useprobs:
                #    labels = torch.argmax(outputs, dim=1)
                #    labels_onehot = make_one_hot(labels, outputs.shape[1])
                #    outputs = labels_onehot
                Dy.append(outputs)
        Dy = torch.cat(Dy)
        #torch.save(blackbox.elbo_cumavg, osp.join(self.out_dir, 'elbo_cumavg.pt'))
        torch.save(self.blackbox.query_dist, osp.join(self.out_dir, "query_dist.pt"))
        torch.save(self.blackbox.conf_adv, osp.join(self.out_dir, "conf_adv.pt"))
        torch.save(self.blackbox.blackbox.conf_vic, osp.join(self.out_dir, "conf_vic.pt"))

        # TensorDataset D
        self.D = TensorDataset(Dx, Dy)

        ### Block memory required for training later on
        model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                           num_classes=self.num_classes)
        model_adv = model_adv.to(self.device)
        _, _, model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                            checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                            device=self.device, epochs=1, ema_decay=self.ema_decay,
                                            log_interval=500, lr=1e-2, momentum=0.9, batch_size=self.batch_size,
                                            lr_gamma=0.1, testset=self.testset,
                                            criterion_train=model_utils.soft_cross_entropy)


    def _init_log(self):
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as log:
                columns = ["Adv_Vic", "Adv_Adv", "Adv_Both"]
                log.write('\t'.join(columns) + '\n')
        print(f"Created adversary log file at {self.log_path}")

    def _write_log(self, msg):
        with open(self.log_path, 'a') as log:
            log.write(msg + '\n')

    def _write_log_table(self, v_a, v, a, total):
        nota = total - a
        notv = total - v
        notv_a = a - v_a
        v_nota = v - v_a
        notv_nota = nota - v_nota

        nota = round(nota / total, 2)
        notv = round(notv / total, 2)
        v_a = round(v_a / total, 2)
        v_nota = round(v_nota / total, 2)
        notv_a = round(notv_a / total, 2)
        notv_nota = round(notv_nota / total, 2)
        v = round(v / total, 2)
        a = round(a / total, 2)
        total /= total
        row0 = '\t'.join(["VA", "A", "notA", "marginV"])
        row1 = '\t'.join(["V", str(v_a), str(v_nota), str(v)])
        row2 = '\t'.join(["notV", str(notv_a), str(notv_nota), str(notv)])
        row3 = '\t'.join(['marginA', str(a), str(nota), str(total)])
        with open(self.log_path, 'a') as log:
            tabel = '\n'.join([row0, row1, row2, row3]) + '\n'+'\n'
            print(tabel)
            log.write(tabel)

    def normalize(self, images):
        mean, std = self.MEAN.to(images.device), self.STD.to(images.device)
        return (images-mean) / std

    def denormalize(self, images):
        mean, std = self.MEAN.to(images.device), self.STD.to(images.device)
        return torch.clamp(images*std + mean, 0., 1.)   

    def get_transferset(self):
        """
        :return:
        """
        # for rho_current in range(self.rho):
        rho_current = 0
        model_adv = None
        if not self.random_adv:
            while self.blackbox.call_count < self.budget and rho_current<self.rho:
                print('=> Beginning substitute epoch {} (|D| = {})'.format(rho_current, len(self.D)))
                # -------------------------- 0. Initialize Model
                model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                        num_classes=self.num_classes)
                #model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, 
                #"/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/vgg16_bn/checkpoint.pth.tar",
                #                        num_classes=self.num_classes)

                model_adv = model_adv.to(self.device)

                # -------------------------- 1. Train model on D
                #model_utils.test_step(model_adv, 
                #          DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=10),
                #          nn.CrossEntropyLoss(reduction='mean'), device=self.device, epoch=0)
                
                ################################################
                # Enable data augmentation
                ################################################
                #Dx, Dy = self.D.tensors                                      
                #transform = None
                #if self.adv_transform:
                #    Dx = self.denormalize(Dx)
                #    #transform = datasets.modelfamily_to_transforms[self.modelfamily]["train_kornia"]
                #    transform = datasets.modelfamily_to_transforms[self.modelfamily]["train"]
                ##self.D = ImageTensorSetKornia((Dx, Dy), transform=transform)
                #self.D = ImageTensorSet((Dx, Dy), transform=transform)
                _, _, model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                                    checkpoint_suffix='.{}'.format(self.blackbox.call_count), ema_decay=self.ema_decay,
                                                    device=self.device, epochs=self.train_epochs, log_interval=500, lr=1e-2,
                                                    momentum=0.9, batch_size=self.batch_size, lr_gamma=0.1,
                                                    testset=self.testset, criterion_train=model_utils.soft_cross_entropy)
                #Dx, Dy = self.D.data, self.D.targets
                #self.D = TensorDataset(Dx, Dy)

                # -------------------------- 2. Evaluate model
                # _, acc = model_utils.test_step(model_adv, self.testloader, nn.CrossEntropyLoss(reduction='mean'),
                #                                device=self.device, epoch=rho_current)
                # self.accuracies.append(acc)

                # -------------------------- 3. Jacobian-based data augmentation
                model_adv.eval()
                if self.aug_strategy in ['jbda', 'jbself']:
                    self.D = self.jacobian_augmentation(model_adv, rho_current, step_size=self.eps, num_steps=self.num_steps)
                elif self.aug_strategy == 'jbtop{}'.format(self.topk):
                    self.D = self.jacobian_augmentation_topk(model_adv, rho_current, step_size=self.eps, num_steps=self.num_steps, batch_size=self.aug_batch_size,
                                                             take_lastk=self.take_lastk, use_foolbox=False, binary_search=self.binary_search)
                    #torch.save(self.blackbox.elbo_cumavg, osp.join(self.out_dir, "elbo_cumavg.pt"))
                    #torch.save(self.blackbox.query_dist, osp.join(self.out_dir, "query_dist.pt"))
                elif self.aug_strategy == 'fbtop{}'.format(self.topk):
                    self.D = self.jacobian_augmentation_topk(model_adv, rho_current, step_size=self.eps, num_steps=self.num_steps, batch_size=self.aug_batch_size,
                                                             take_lastk=self.take_lastk, use_foolbox=True, use_feature_fool=self.use_feature_fool)
                else:
                    raise ValueError('Unrecognized augmentation strategy: "{}"'.format(self.aug_strategy))

                torch.save(self.blackbox.conf_adv, osp.join(self.out_dir, "conf_adv.pt"))
                torch.save(self.blackbox.blackbox.conf_vic, osp.join(self.out_dir, "conf_vic.pt"))
                # -------------------------- 4. End if necessary
                rho_current += 1
                torch.save(self.D, osp.join(self.out_dir, "transferset.pt"))
                print(f"Save transferset to {osp.join(self.out_dir, 'transferset.pt')}")
                #torch.save(self.blackbox.query_dist, osp.join(self.out_dir, "query_dist.pt"))

                if (self.blackbox.call_count >= self.budget) or ((self.rho is not None) and (rho_current >= self.rho)):
                    print('=> # BB Queries ({}) >= budget ({}). Ending attack.'.format(self.blackbox.call_count,
                                                                                       self.budget))
                    model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                            num_classes=self.num_classes)
                    model_adv = model_adv.to(self.device)

                    ################################################
                    # Enable data augmentation
                    ################################################
                    Dx, Dy = self.D.tensors                                      
                    transform = None
                    if self.adv_transform:
                        Dx = self.denormalize(Dx)
                        transform = datasets.modelfamily_to_transforms[self.modelfamily]["train"]
                        #transform = datasets.modelfamily_to_transforms[self.modelfamily]["train_kornia"]
                    #self.D = ImageTensorSetKornia((Dx, Dy), transform=transform)
                    self.D = ImageTensorSet((Dx, Dy), transform=transform)

                    _, _, model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                                        checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                                        device=self.device, epochs=self.final_train_epochs, ema_decay=self.ema_decay,
                                                        log_interval=500, lr=1e-2, momentum=0.9, batch_size=self.batch_size,
                                                        lr_gamma=0.1, testset=self.testset,
                                                        criterion_train=model_utils.soft_cross_entropy)
                    break

                print()
        else:
            torch.save(self.blackbox.blackbox.conf_vic, osp.join(self.out_dir, "conf_vic.pt"))
            exit(0)
            print('=> # BB Queries ({}) >= budget ({}). Ending attack.'.format(self.blackbox.call_count,
                                                                               self.budget))
            model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                    num_classes=self.num_classes)
            model_adv = model_adv.to(self.device)

            ################################################
            # Enable data augmentation
            ################################################
            Dx, Dy = self.D.tensors                                      
            transform = None
            if self.adv_transform:
                Dx = self.denormalize(Dx)
                transform = datasets.modelfamily_to_transforms[self.modelfamily]["train"]
            self.D = ImageTensorSet((Dx, Dy), transform=transform)

            _, _, model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                                checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                                device=self.device, epochs=self.final_train_epochs,
                                                log_interval=500, lr=0.01, momentum=0.9, batch_size=self.batch_size,
                                                lr_gamma=0.1, testset=self.testset,
                                                criterion_train=model_utils.soft_cross_entropy)

        return self.D, model_adv


    @staticmethod
    def rand_sample(D, kappa):
        # Note: the paper does reservoir sampling to select kappa elements from D. Since |D| in our case cannot grow
        # larger than main memory size, we randomly sample for simplicity. In either case, each element is drawn with a
        # probability kappa/|D|
        n = len(D)
        idxs = np.arange(n)
        sampled_idxs = np.random.choice(idxs, size=kappa, replace=False)
        mask = np.zeros_like(idxs).astype(bool)
        mask[sampled_idxs] = True
        D_sampled = TensorDataset(D.tensors[0][mask], D.tensors[1][mask])
        return D_sampled

    @staticmethod
    def sample_lastk(D, lastk):
        n = len(D)
        D_sampled = TensorDataset(D.tensors[0][:lastk], D.tensors[1][:lastk])
        return D_sampled

    def jacobian_augmentation(self, model_adv, rho_current, step_size=0.1, num_steps=8):
        if (self.kappa is not None) and (rho_current >= self.sigma):
            D_sampled = self.rand_sample(self.D, self.kappa)
        else:
            D_sampled = self.D

        if len(D_sampled) + self.blackbox.call_count >= self.budget:
            # Reduce augmented data size to match query budget
            nqueries_remaining = self.budget - self.blackbox.call_count
            assert nqueries_remaining >= 0
            print('=> Reducing augmented input size ({} -> {}) to stay within query budget.'.format(
                D_sampled.tensors[0].shape[0], nqueries_remaining))
            D_sampled = TensorDataset(D_sampled.tensors[0][:nqueries_remaining],
                                      D_sampled.tensors[1][:nqueries_remaining])

        if self.tau is not None:
            step_size = step_size * ((-1) ** (round(rho_current / self.tau)))

        print('=> Augmentation set size = {} (|D| = {}, B = {})'.format(len(D_sampled), len(self.D),
                                                                        self.blackbox.call_count))
        loader = DataLoader(D_sampled, batch_size=self.query_batch_size, shuffle=False)
        for i, (X, Y) in enumerate(loader):
            start_idx = i * self.query_batch_size
            end_idx = min(start_idx + self.query_batch_size, len(D_sampled))
            # A simple check to ensure we are overwriting the correct input-outputs
            assert Y.sum() == D_sampled.tensors[1][start_idx:end_idx].sum(), '[{}] {} != {}'.format(i, Y.sum(),
                                                                                                    D_sampled.tensors[
                                                                                                        1][
                                                                                                    start_idx:end_idx].sum())
            assert X.sum() == D_sampled.tensors[0][start_idx:end_idx].sum(), '[{}] {} != {}'.format(i, X.sum(),
                                                                                                    D_sampled.tensors[
                                                                                                        0][
                                                                                                    start_idx:end_idx].sum())

            # Get augmented inputs
            X, Y = X.to(self.device), Y.to(self.device)
            delta_i = self.fgsm_untargeted(model_adv, X, Y.argmax(dim=1), device=self.device, epsilon=step_size)
            # Get corrensponding outputs from blackbox
            if self.aug_strategy == 'jbda':
                Y_i = self.blackbox(X + delta_i)
            elif self.aug_strategy == 'jbself':
                Y_i = self.blackbox(X - delta_i)
            else:
                raise ValueError('Unrecognized augmentation strategy {}'.format(self.aug_strategy))

            #if not self.useprobs:
            #    labels = torch.argmax(Y_i, dim=1)
            #    labels_onehot = make_one_hot(labels, Y_i.shape[1])
            #    Y_i = labels_onehot

            # Rewrite D_sampled
            D_sampled.tensors[0][start_idx:end_idx] = (X + delta_i).detach().cpu()
            D_sampled.tensors[1][start_idx:end_idx] = Y_i.detach().cpu()

        Dx_augmented = torch.cat([self.D.tensors[0], D_sampled.tensors[0]])
        Dy_augmented = torch.cat([self.D.tensors[1], D_sampled.tensors[1]])
        D_augmented = TensorDataset(Dx_augmented, Dy_augmented)

        return D_augmented

    def jacobian_augmentation_topk(self, model_adv, rho_current, step_size=0.1, num_steps=8, batch_size=256, 
                                         take_lastk=-1, use_foolbox=False, binary_search=False, use_feature_fool=False):
        if (self.kappa is not None) and (rho_current >= self.sigma):
            D_sampled = self.rand_sample(self.D, self.kappa)
        elif take_lastk > 0:
            D_sampled = self.sample_lastk(self.D, take_lastk)
        else:
            D_sampled = self.D

        if (len(D_sampled) * self.topk) + self.blackbox.call_count >= self.budget:
            # Reduce augmented data size to match query budget
            nqueries_remaining = self.budget - self.blackbox.call_count
            nqueries_remaining /= self.topk #3.
            nqueries_remaining = int(np.ceil(nqueries_remaining))
            assert nqueries_remaining >= 0

            try:
                print('=> Reducing augmented input tensors[1]size ({}*{} -> {}*{}={}) to stay within query budget.'.format(
                    D_sampled.tensors[0].shape[0], self.topk, nqueries_remaining, self.topk,
                    nqueries_remaining * self.topk))
                D_sampled = TensorDataset(D_sampled.tensors[0][:nqueries_remaining],
                                          D_sampled.tensors[1][:nqueries_remaining])
            except AttributeError as e:
                print('=> Reducing augmented input tensors[1]size ({}*{} -> {}*{}={}) to stay within query budget.'.format(
                    D_sampled.data.shape[0], self.topk, nqueries_remaining, self.topk,
                    nqueries_remaining * self.topk))
                D_sampled = TensorDataset(D_sampled.data[:nqueries_remaining],
                                          D_sampled.targets[:nqueries_remaining])

        if self.tau is not None:
            step_size = step_size * ((-1) ** (round(rho_current / self.tau)))

        print('=> Augmentation set size = {} (|D| = {}, B = {})'.format(len(D_sampled), len(self.D),
                                                                        self.blackbox.call_count))

        if binary_search or not use_foolbox:
            batch_size = 1
        loader = DataLoader(D_sampled, batch_size=batch_size, shuffle=True)
        X_aug = []
        Y_aug = []
        adv2bb = 0
        adv2adv = 0
        adv2both = 0
        adv2bb_t = 0
        adv2adv_t = 0
        adv2both_t = 0
        num_skips = 0
        asr = 0. # attack success rate
        fb_is_adv = None
        total = 0.

        if use_foolbox or binary_search:
            model_adv.eval()
            fmodel = foolbox.models.PyTorchModel(model_adv, bounds=(0, 1), preprocessing={"mean":self.MEAN.to(self.device), "std":self.STD.to(self.device)})
            #fmodel = foolbox.models.PyTorchModel(self.blackbox.blackbox.model, bounds=(0, 1), preprocessing={"mean":self.MEAN.to(self.device), "std":self.STD.to(self.device)})
        """
        Note: Attacks by Foolbox supports batch attacks, BUT remaining budget 
              may not be counted accurately.
        """
        for i, (X, Y) in enumerate(loader):
            #if not (use_foolbox or binary_search):
            #    assert X.shape[0] == Y.shape[0] == 1, 'JB-top3 only supports batch_size = 1'
            X, Y = X.to(self.device), Y.to(self.device)
            total += X.size(0)*self.topk
            # --------------------- Blackbox prediction before augmentaion 
            with torch.no_grad():
                adv_before = model_adv(X).argmax(-1)
                try: # Using detector
                    bb_before = self.blackbox.blackbox(X).argmax(-1)
                except AttributeError as e:
                    bb_before = self.blackbox(X).argmax(-1)

            with torch.no_grad():
                Y_pred = model_adv(X)

            #if not (use_foolbox or binary_search):
            #    Y_pred_sorted = torch.argsort(Y_pred[0], descending=True)
            #    Y_pred_sorted = Y_pred_sorted[Y_pred_sorted != Y[0].argmax()]  # Remove gt class
            #else:
            Y_pred_sorted = torch.argsort(Y_pred, descending=True)
            new_Y_pred_sorted = []
            for i, y in enumerate(Y_pred_sorted):
                new_Y_pred_sorted.append(y[y != Y[i].argmax()]) # Remove gt class
            Y_pred_sorted = torch.stack(new_Y_pred_sorted)
            for ci in range(self.topk):
                if len(Y_pred_sorted.shape) == 1:
                    c = Y_pred_sorted[ci]
                elif len(Y_pred_sorted.shape) == 2: 
                    c = Y_pred_sorted[:, ci]
                else:
                    raise RuntimeError
                if use_foolbox:
                    if use_feature_fool:
                        delta_i, fb_is_adv = self.feature_fool(fmodel, model_adv, X, Y.argmax(dim=1), c, epsilon=step_size, alpha=0.01, device=self.device, num_iter=25)
                    else:
                        delta_i, fb_is_adv = self.foolbox_targ(fmodel, X, Y.argmax(dim=1), c, epsilon=step_size, alpha=0.01,
                                                     device=self.device, attack_alg=self.foolbox_alg)

                elif binary_search:
                    #delta_i, fb_is_adv = self.binary_search_linf_targ(fmodel, self.blackbox, X, Y.argmax(dim=1), c,
                    #                              epsilon=step_size, alpha=0.01, device=self.device)
                    delta_i, fb_is_adv = self.binary_search_linf_targ(fmodel, model_adv, X, Y.argmax(dim=1), c,
                                                  epsilon=step_size, alpha=0.01, device=self.device)
                else:
                    delta_i = self.pgd_linf_targ(model_adv, X, Y.argmax(dim=1), c, epsilon=step_size, alpha=0.01,
                                             device=self.device)

                if delta_i is None:
                    num_skips += 1
                    continue

                asr += fb_is_adv.to(torch.float32).sum().item() if fb_is_adv is not None else 0

                x_aug = X + delta_i
                """
                Adversarial Detector
                """
                if self.detector_adv is not None:
                    size_before = x_aug.size(0)
                    is_adv = self.detector_adv(x_aug)
                    selected_idx = (torch.Tensor(is_adv) == False).to(self.device)
                    x_aug = x_aug[selected_idx]
                    size_after = x_aug.size(0)
                    print(f"Filter out {size_before-size_after}/{size_before}")
                    num_skips += size_before-size_after
                    if size_after == 0:
                        continue
                with torch.no_grad():
                    Y_adv = model_adv(x_aug)
                    conf_a, Y_adv_pred = F.softmax(Y_adv, dim=-1).max(-1)
                    self.blackbox.conf_adv.append(conf_a.cpu().numpy())
                if self.blinder_fn is not None:
                    x_query = self.blinder_fn(x_aug)
                else:
                    x_query = x_aug
                try: # No detector
                    if isinstance(self.blackbox, ELBODetector2):
                        Y_i = self.blackbox(x_query, Y.argmax(dim=1))
                    else:
                        Y_i = self.blackbox(x_query, is_adv=True)
                except TypeError as e:
                    Y_i = self.blackbox(x_query)
                #if self.blackbox.output_type == "logits":
                #    conf_v, Y_i_pred = F.softmax(Y_i, dim=-1).max(-1)
                #elif self.blackbox.output_type == "probs":
                #    conf_v, Y_i_pred = Y_i.max(-1)
                #else:
                #    print("Warning: ")
                Y_i_pred = Y_i.argmax(-1)

                # Compare predictions
                # Untargeted (simply changing labels)
                adv2bb_i = bb_before.ne(Y_i_pred)#.sum().item()
                adv2adv_i = adv_before.ne(Y_adv_pred)#.sum().item()
                adv2both_i =  torch.logical_and(adv2bb_i ,adv2adv_i)

                adv2bb += adv2bb_i.sum().item()
                adv2adv += adv2adv_i.sum().item()
                adv2both += adv2both_i.sum().item()

                # Targeted
                adv2bb_it = Y_i_pred.eq(c)#.sum().item()
                adv2adv_it = Y_adv.argmax(-1).eq(c)#.sum().item()
                adv2both_it =  torch.logical_and(adv2bb_it ,adv2adv_it)

                adv2bb_t += adv2bb_it.sum().item()
                adv2adv_t += adv2adv_it.sum().item()
                adv2both_t += adv2both_it.sum().item()

                #if not self.useprobs:
                #    labels = torch.argmax(Y_i, dim=1)
                #    labels_onehot = make_one_hot(labels, Y_i.shape[1])
                #    Y_i = labels_onehot

                #X_aug.append(X.detach().cpu().clone())
                X_aug.append(x_aug.detach().cpu().clone())
                Y_aug.append(Y_i.detach().cpu().clone())

            if self.blackbox.call_count >= self.budget:
                break

        print(f"skip: {num_skips}/{total}")
        print(f"attack success rate: {asr/total*100}%")
        if len(X_aug) > 0:        
            X_aug = torch.cat(X_aug, dim=0)
            Y_aug = torch.cat(Y_aug, dim=0)
        else:
            return D_sampled

        try: 
            Dx_augmented = torch.cat([self.D.tensors[0], X_aug])[:self.budget]
            Dy_augmented = torch.cat([self.D.tensors[1], Y_aug])[:self.budget]
        except AttributeError as e:
            Dx_augmented = torch.cat([self.D.data, X_aug])[:self.budget]
            Dy_augmented = torch.cat([self.D.targets, Y_aug])[:self.budget]

        D_augmented = TensorDataset(Dx_augmented, Dy_augmented)

        #------------------------- Logging
        # Write number of adversarial examples to log

        print("Untargeted:")
        msg = f"{adv2bb}/{total} ({int(100*adv2bb/total)})%"
        print(msg+" are adversarial to the victim")
        self._write_log(msg+" are adversarial to the victim")

        msg2 = f"{adv2adv}/{total} ({int(100*adv2adv/total)})%"
        print(msg2+" are adversarial to the adversary")
        self._write_log(msg2+" are adversarial to the adversary")

        msg3 = f"{adv2both}/{total} ({int(100*adv2both/total)})%"
        print(msg3+" are adversarial to both")
        self._write_log(msg3+" are adversarial to both")

        if self.log_path:
            self._write_log("Untargeted:")
            self._write_log("\t".join([msg, msg2, msg3]))
            self._write_log_table(adv2both, adv2bb, adv2adv, total)

        print()
        print("Targeted:")
        msg = f"{adv2bb_t}/{total} ({int(100*adv2bb_t/total)})%"
        print(msg+" are adversarial to the victim")
        msg2 = f"{adv2adv_t}/{total} ({int(100*adv2adv_t/total)})%"
        print(msg2+" are adversarial to the adversary")
        msg3 = f"{adv2both_t}/{total} ({int(100*adv2both_t/total)})%"
        print(msg3+" are adversarial to both")
        if self.log_path:
            self._write_log("Targeted:")
            self._write_log("\t".join([msg, msg2, msg3]))
            self._write_log_table(adv2both_t, adv2bb_t, adv2adv_t, total)
            self._write_log(f"skip: {num_skips}/{total}")
            self._write_log(f"ASR by Foolbox: {asr/total*100}%")
        return D_augmented

    @staticmethod
    def fgsm_untargeted(model, inputs, targets, epsilon, device, clamp=(0.0, 1.0)):
        if epsilon == 0:
            return torch.zeros_like(inputs)

        with torch.enable_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True

            out = model(inputs)
            loss = F.cross_entropy(out, targets)
            loss.backward()

            delta = epsilon * inputs.grad.detach().sign().to(device)

            delta.data = torch.min(torch.max(delta, -inputs),
                                   1 - inputs)  # clip samples+perturbation to [0,1]

            return delta

    @staticmethod
    def pgd_linf_targ(model, inputs, targets, y_targ, epsilon, alpha, device, num_iter=8):
        """ Construct targeted adversarial examples on the examples X"""
        if epsilon == 0:
            return torch.zeros_like(inputs)

        #alpha = 2*epsilon/num_iter
        alpha = 0.01
        with torch.enable_grad():
            inputs = inputs.to(device)
            delta = torch.zeros_like(inputs, requires_grad=True).to(device)
            for t in range(num_iter):
                yp = model(inputs + delta)
                # loss = (yp[:, y_targ] - yp.gather(1, targets[:, None])[:, 0]).sum()
                loss = yp[:, y_targ].sum()
                loss.backward()
                delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
                delta.grad.zero_()
            return delta.detach()

    def foolbox_targ(self, fmodel, inputs, targets, y_targ, epsilon, alpha, device, num_iter=8, attack_alg="pgd"):
        if attack_alg=="pgd":
            #attack = PGD(abs_stepsize=2*epsilon/num_iter)
            adv_criterion = TargetedMisclassification(y_targ)
            attack = PGD()
            eps = 8./256
        elif attack_alg=="cw_l2":
            #adv_criterion = Misclassification(targets)
            adv_criterion = TargetedMisclassification(y_targ)
            attack = L2CarliniWagnerAttack(steps=150, binary_search_steps=5)
            eps = None
        else:
            raise ValueError("Supported attack alg: ['pgd', 'cw_l2']")
        #print(f"Start attack by {attack}...")
        inputs = inputs.to(device)

        images = self.denormalize(inputs)
        #images = inputs.clone()
        #fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), preprocessing={"mean":self.MEAN.to(device), "std":self.STD.to(device)})
        #fmodel = foolbox.models.PyTorchModel(model, bounds=(-2.5, 2.8))
        _, images, is_adv = attack(fmodel, images, criterion=adv_criterion, epsilons=eps)
        images = self.normalize(images)
        print(f"avg l2-dist: {foolbox.distances.l2(images, inputs).mean()}")

        delta = images - inputs
        return delta, is_adv

    def binary_search_linf_targ(self, fmodel, model_vic, inputs, targets, y_targ, epsilon, alpha, device, num_iter=8, threshold=1e-6):
        attack = PGD()
        inputs = inputs.to(device)

        images = self.denormalize(inputs)
        #images = inputs.clone()
        adv_criterion = TargetedMisclassification(y_targ)
        #fmodel = foolbox.models.PyTorchModel(model_adv, bounds=(0, 1), preprocessing={"mean":self.MEAN.to(device), "std":self.STD.to(device)})
        #fmodel = foolbox.models.PyTorchModel(model, bounds=(-2.5, 2.8))
        _, images, is_adv = attack(fmodel, images, criterion=adv_criterion, epsilons=8./256)
        images = self.normalize(images)

        if is_adv.item():
            images_b = self.boundary_search(model_vic, images.clone(), y_targ, inputs.clone(), targets, thresh=threshold)
            #images_b = self.boundary_search(model_adv, images.clone(), y_targ, inputs.clone(), targets, thresh=threshold)
            delta = images_b - inputs
            #print(torch.dist(images, inputs, 2))
            #print(torch.dist(images_b, inputs, 2))
            #print(torch.dist(images_b, images, 2))
            #with torch.no_grad():
            ##    #adv_feat_b = self.detector_adv.encoder(images_b)
            ##    #adv_feat = self.detector_adv.encoder(images)

            #    vic_feat_b = self.blackbox.encoder(images_b)
            #    vic_feat = self.blackbox.encoder(images)
            #    vic_feat_ori = self.blackbox.encoder(inputs)

            ##    #print(torch.dist(adv_feat_b, adv_feat, 2))
            ##    print(torch.dist(vic_feat, vic_feat_ori, 2))
            ##    print(torch.dist(vic_feat_b, vic_feat_ori, 2))
            ##    print(torch.dist(vic_feat_b, vic_feat, 2))
            ##import ipdb; ipdb.set_trace()
            #self.images2inputs.append(torch.dist(images, inputs, 2).item())
            #self.imagesb2inputs.append(torch.dist(images_b, inputs, 2).item())

            #self.feat2feat_org.append(torch.dist(vic_feat, vic_feat_ori, 2).item()) 
            #self.feat2feat_b.append(torch.dist(vic_feat_b, vic_feat_ori, 2).item())

            #torch.save(self.images2inputs, "./images2inputs.pylist")
            #torch.save(self.imagesb2inputs, "./images2binputs.pylist")
            #torch.save(self.feat2feat_org, "./feat2feat_org.pylist")
            #torch.save(self.feat2feat_b, "./feat2feat_b.pylist")
        else:
            return None, is_adv.item()
        return delta, is_adv

    def boundary_search(self, model, x1, y1, x2, y2, thresh, max_attempts=1000):
        attempts = 0
        with torch.no_grad():
            while True:

                mid = (x1+x2)/2
                attempts += 1                        

                if attempts > max_attempts:
                    return mid

                if self.blackbox.call_count >= self.budget:
                    return mid

                Y = model(mid)
                pred_mid = torch.argmax(Y, dim=1)

                if (pred_mid != y1).item() and (pred_mid != y2).item():
                    return mid
                elif torch.max(torch.abs(x1-x2)).item() < thresh:
                    return mid
                elif pred_mid == y1:
                    x1 = mid
                elif pred_mid == y2:
                    x2 = mid                    

    def feature_fool(self, fmodel, model, inputs, targets, y_targ, epsilon, alpha, device, num_iter=8):
        """ Construct targeted adversarial examples on the examples X"""
        attack = PGD()
        inputs = inputs.to(device)

        images = self.denormalize(inputs)
        #images = inputs.clone()
        adv_criterion = TargetedMisclassification(y_targ)
        #fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), preprocessing={"mean":self.MEAN.to(device), "std":self.STD.to(device)})
        #fmodel = foolbox.models.PyTorchModel(model, bounds=(-2.5, 2.8))
        _, images, is_adv = attack(fmodel, images, criterion=adv_criterion, epsilons=8./256)
        images = self.normalize(images)

        #images, inputs = images[is_adv], inputs[is_adv]
        batch_size = inputs.size(0)
        if batch_size > 0:
            num_iter = 100
            #alpha = 2*epsilon/num_iter
            alpha = 1e-4
            lam = 1.
            source_input, target_input = inputs, images

            with torch.enable_grad():
                delta = torch.zeros_like(inputs, requires_grad=True).to(device)
                source_input, target_input = source_input.to(device), target_input.to(device)

                target_feature = model.features(target_input)
                target_feature = target_feature.view(target_feature.size(0), -1)

                source_feature = model.features(source_input)
                source_feature = source_feature.view(source_feature.size(0), -1)

                # TODO: calculate margin
                offset = torch.cdist(source_feature, source_feature).sum() / (batch_size*(batch_size-1))
                triplet_loss = torch.nn.TripletMarginLoss(margin=0.5-offset.item())
                for _ in range(num_iter):
                    delta_feature= model.features(source_input+delta)

                    triplet = triplet_loss(delta_feature, target_feature, source_feature)
                                            # anchor         # positive      # negative
                    delta_norm = torch.norm(delta)
                    loss = delta_norm + lam*triplet

                    loss.backward(retain_graph=True)
                    #delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
                    delta.data = (delta - alpha * delta.grad.detach().sign())
                    delta.data = (source_input+delta).clamp(-2.5, 2.8) - source_input
                    delta.grad.zero_()
#                    print(loss.item(), delta.max().item(), delta_norm.item(), lam*triplet.item())
#                print()
            return delta.detach(), is_adv
        else:
            return None, is_adv