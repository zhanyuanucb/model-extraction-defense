import torchvision.models as models
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp
import pickle
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset, DataLoader

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
#from attack.adversary.adv import*
import pretrained
def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

model_path = '/mydata/model-extraction/model-extraction-defense/models/checkpoint.budget100.pth.tar'
model = models.resnet50()
if gpu_count > 1:
   model = nn.DataParallel(model)
model = model.to(device)

optimizer_name = "adam"
optimizer = get_optimizer(model.parameters(), optimizer_name)
queryset_name = "ImageNet1k"
modelfamily = datasets.dataset_to_modelfamily[queryset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test']

if osp.isfile(model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(model_path))

dataset_name = "ImageNet1k"
dataset = datasets.__dict__[dataset_name]
testset = dataset(train=False, transform=transform)
batch_size = 128
num_workers = 10
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=None)

tic = time.time()
model_utils.test_step(model, test_loader, criterion_test, device)
tac = time.time()
print("validation time: {} min".format((tac - tic)/60))