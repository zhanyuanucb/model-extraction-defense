import os
import os.path as osp
import attack.config as cfg
from attack import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import shutil

dataset_dir = osp.join(cfg.DATASET_ROOT, "cifar10/train") 
dest_root = osp.join("/mydata/model-extraction/data/cifar10_balanced_subset6400")
if not osp.exists(dest_root):
    os.mkdir(dest_root)

SIZE = 640
for c in os.listdir(dataset_dir):
    # sample SIZE images from each class
    src_dir = osp.join(dataset_dir, c)
    image_lst = os.listdir(src_dir)
    sampled_idxs = np.random.choice(list(range(len(image_lst))), replace=False, size=SIZE)
    sampled_images = np.array(image_lst)[sampled_idxs]
    dest_dir = osp.join(dest_root, c)
    if not osp.exists(dest_dir):
        os.mkdir(dest_dir)
    for image in sampled_images:
        # copy each image to the destination
        from_dir = osp.join(src_dir, image)
        to_dir = osp.join(dest_dir, image)
        shutil.copy(from_dir, to_dir)