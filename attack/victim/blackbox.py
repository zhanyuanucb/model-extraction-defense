#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import json
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
#sys.path.append('/mydata/model-extraction/knockoffnets')

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from attack.utils.type_checks import TypeCheck
import attack.utils.model as model_utils
import modelzoo.zoo as zoo
from attack import datasets

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
__status__ = "Development"

gpu_count = torch.cuda.device_count()

class Blackbox(object):
    def __init__(self, model, device=None, output_type="one_hot", T=1):
        self.model = model.to(device)
        self.model.eval()
        self.output_type = output_type # ["one_hot", "logits", "prob"]
        self.T = T
        print(f"############## T={self.T} ################")
        self.call_count = 0
        self.device = device

    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type="one_hot", T=1):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        victim_dataset = params.get('dataset', 'imagenet')
        modelfamily = datasets.dataset_to_modelfamily[victim_dataset]

        # Instantiate the model
        model = zoo.get_net(model_arch, modelfamily, pretrained=None, num_classes=num_classes)
        model = model.to(device)
        model.eval()

        # Load weights
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        #import ipdb; ipdb.set_trace()
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model, output_type=output_type, T=T)
        blackbox.device = device
        return blackbox

    def __call__(self, images, is_adv=False):
        if is_adv:
            self.call_count += images.size(0)
        images = images.to(self.device)
        with torch.no_grad():
            logits = self.model(images)
        if self.output_type == "logits":
            return logits

        topk_vals, indices = torch.topk(logits, 1)
        y = torch.zeros_like(logits)

        if self.output_type == "one_hot":
            return y.scatter(1, indices, torch.ones_like(topk_vals))
        #return F.softmax(logits, dim=1)
        p = F.softmax(logits, dim=1)
        return F.softmax(p.pow(1/self.T), dim=1)

    def eval(self):
        self.model.eval()

    def to(self, device):
        return self.model.to(device)