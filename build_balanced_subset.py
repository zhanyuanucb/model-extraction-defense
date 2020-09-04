import os
import os.path as osp
import attack.config as cfg
from attack import datasets
from attack.utils.utils import create_dir
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import shutil

#dataset_dir = osp.join(cfg.DATASET_ROOT, "CINIC10_2/train") 
#dest_root = osp.join("/mydata/model-extraction/data/cinic10_balanced_subset65000")
dataset_dir = osp.join(cfg.DATASET_ROOT, "cifar10/train") 
dest_root = osp.join("/mydata/model-extraction/data/cifar10_balanced_subset35000")
create_dir(dest_root)

np.random.seed(cfg.DS_SEED)
SIZE = 3500

for c in os.listdir(dataset_dir):
    # sample SIZE images from each class
    src_dir = osp.join(dataset_dir, c)
    image_lst = os.listdir(src_dir)
    sampled_idxs = np.random.choice(list(range(len(image_lst))), replace=False, size=SIZE)
    sampled_images = np.array(image_lst)[sampled_idxs]
    dest_dir = osp.join(dest_root, c)
    create_dir(dest_dir)
    for image in sampled_images:
        # copy each image to the destination
        from_dir = osp.join(src_dir, image)
        to_dir = osp.join(dest_dir, image)
        shutil.copy(from_dir, to_dir)