import os
import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import attack.config as cfg


class CINIC10(ImageFolder):
    def __init__(self, split="train", transform=None, target_transform=None):
        assert split in ["train", "valid", "test"]
#        if train:
#            root = osp.join(cfg.DATASET_ROOT, "CINIC10_2", "train")
#        else:
#            root = osp.join(cfg.DATASET_ROOT, "CINIC10_2", "valid")
        root = osp.join(cfg.DATASET_ROOT, "CINIC10_2", split)
        super().__init__(root, transform=transform, target_transform=target_transform)