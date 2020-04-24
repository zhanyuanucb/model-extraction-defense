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
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_modeldir(cls, model_dir, device=None):
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
        # model = model_utils.get_net(model_arch, n_output_classes=num_classes)
        model = zoo.get_net(model_arch, modelfamily, pretrained=None, num_classes=num_classes)
        # TODO: Multi-GPU 
        #if gpu_count > 1:
        #    model = nn.DataParallel(model)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model)
        return blackbox

    def __call__(self, images):
        images = images.cuda()
        with torch.no_grad():
            logits = self.model(images)
        topk_vals, indices = torch.topk(logits, 1)
        y = torch.zeros_like(logits)
        return y.scatter(1, indices, torch.ones_like(topk_vals))
        #return F.softmax(logits, dim=1)