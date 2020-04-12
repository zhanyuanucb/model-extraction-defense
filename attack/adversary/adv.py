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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.seedset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))

    def get_seedset(self):
        remain = self.n_queryset
        images, labels = [], []
        with tqdm(total=self.n_queryset) as pbar:
            for t, B in enumerate(range(0, self.n_queryset, self.batch_size)):
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, remain))
                self.idx_set = self.idx_set - set(idxs)
                remain -= len(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.queryset)))

                #x_t = torch.stack([self.queryset[i][0][0][None] for i in idxs]).cuda() # Only for MNIST
                queryset = self.queryset
                x_t = torch.stack([queryset[i][0] for i in idxs], dim=0)

                x_t = x_t.cuda()
                y_t = self.blackbox(x_t).cpu()
                #if hasattr(self.queryset, 'samples'):
                #    # Any DatasetFolder (or subclass) has this attribute
                #    # Saving image paths are space-efficient
                #    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                #else:
                #    # Otherwise, store the image itself
                #    # But, we need to store the non-transformed version
                #    img_t = [self.queryset.data[i] for i in idxs]
                #    if isinstance(self.queryset.data[0], torch.Tensor):
                #        img_t = [x.numpy() for x in img_t]
                images.append(x_t.cpu().clone())
                labels.append(y_t.clone())
                pbar.update(x_t.size(0))

        self.seedset = [torch.cat(images), torch.cat(labels)]
        return self.seedset


class JDAAdversary(object):
    def __init__(self, adversary_model, blackbox, eps=0.1, batch_size=8, steps=1, momentum=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.JDA = MultiStepJDA(self.adversary_model, self.blackbox, eps=eps, batchsize=batch_size, steps=steps, momentum=momentum)

    def augment(self, dataloader, outdir):
        return self.JDA(dataloader, outdir)