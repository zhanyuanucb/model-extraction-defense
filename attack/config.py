import os
import os.path as osp
from os.path import dirname, abspath
import numpy as np
import random
import torch

#DEFAULT_SEED = 42
#DS_SEED = 123  # uses this seed when splitting datasets
#
#random.seed(DEFAULT_SEED)
#np.random.seed(DEFAULT_SEED)
#torch.manual_seed(DEFAULT_SEED)
#torch.cuda.manual_seed(DEFAULT_SEED)
#torch.backends.cudnn.deterministic = True

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
DATASET_ROOT = "/data"
CIFAR10_DATASET_TRAIN = osp.join(DATASET_ROOT, 'cifar10/training.pt')
dataset2dir = {"CIFAR10": {"train":osp.join(DATASET_ROOT, 'cifar10/training.pt'),
                               "test":osp.join(DATASET_ROOT, 'cifar10/test.pt')},
                    "MNIST": {"train":osp.join(DATASET_ROOT, 'mnist/MNIST/processed/training.pt'),
                              "test":osp.join(DATASET_ROOT, 'mnist/MNIST/processed/test.pt')}
                  }
DEBUG_ROOT = osp.join(PROJECT_ROOT, 'debug')
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')

# -------------- URLs
ZOO_URL = 'http://datasets.d2.mpi-inf.mpg.de/blackboxchallenge'

# -------------- Dataset Stuff
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404] 
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]
MNIST_MEAN = (0.1307,) 
MNIST_STD = (0.3081,)
DEFAULT_BATCH_SIZE = 64

NORMAL_PARAMS = {"cifar": (CIFAR_MEAN, CIFAR_STD),
                 "cinic10": (CINIC_MEAN, CINIC_STD),
                 "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
                 "mnist": (MNIST_MEAN, MNIST_STD)}