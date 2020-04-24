import argparse
import torchvision.models as models
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp
import pickle
from datetime import datetime
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset, DataLoader

import modelzoo.zoo as zoo
import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='Validate model')
    parser.add_argument('--ckp_path', metavar='PATH', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--ckp_suffix', metavar='TYPE', type=str, help='checkpoint suffix')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of validation dataset')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="simnet")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes')
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=128)
    parser.add_argument('--num_workers', metavar='TYPE', type=int, help='Number of processes of dataloader', default=10)

    args = parser.parse_args()
    params = vars(args)

    ckp_path = params['ckp_path']
    ckp_suffix = params['ckp_suffix']
    model_name = params['model_name']
    num_classes = params['num_classes']
    dataset_name = params['dataset_name']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    if gpu_count > 1:
       model = nn.DataParallel(model)
    model = model.to(device)

    transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    model_path = osp.join(ckp_path, f"checkpoint{ckp_suffix}.pth.tar")
    if osp.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        exit(1)

    
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=None)

    # Reference: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    plt.figure(figsize=(8, 6))
    
    confusion = np.array([[0 for _ in range(num_classes)] for _ in range(num_classes)])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #print(f"testing batch {batch_idx}")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            predicted = predicted.cpu().numpy()
            targets = targets.cpu().numpy()
            for y, y_hat in zip(targets, predicted):
                confusion[y][y_hat] += 1
    confusion = confusion.astype('float')
    confusion /= confusion.sum(axis=1)[:, np.newaxis]
    cmap = plt.get_cmap('Blues')
    plt.imshow(confusion, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    thresh = confusion.max() / 1.5
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, "{:0.3f}".format(confusion[i, j]),
                 horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}%'.format(best_test_acc))
    plt.savefig(osp.join(ckp_path, "confusion_matrix.png")) 

if __name__ == '__main__':
    main()