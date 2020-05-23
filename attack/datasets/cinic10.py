import os
import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import attack.config as cfg


class CINIC10(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        if train:
            root = osp.join(cfg.DATASET_ROOT, "CINIC10_2", "train")
        else:
            root = osp.join(cfg.DATASET_ROOT, "CINIC10_2", "valid")
        super().__init__(root, transform=transform, target_transform=target_transform)