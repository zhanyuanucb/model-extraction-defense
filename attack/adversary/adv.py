import sys
import pickle
sys.path.append('/mydata/model-extraction/model-extraction-defense/attack/adversary')
from jda import*
import attack.config as cfg
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as tvdatasets
from attack import datasets

class RandomAdversary(object):
    def __init__(self, blackbox, queryset):
        self.blackbox = blackbox
        self.queryset = queryset

    def get_seedset(self):
        images, labels = [], []
        for x_t, _ in self.queryset:
            x_t = x_t.cuda()
            y_t = self.blackbox(x_t).cpu()
            images.append(x_t.cpu().clone())
            labels.append(y_t.clone())

        seedset = [torch.cat(images), torch.cat(labels)]
        return seedset
