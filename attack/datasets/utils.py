# Reference: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/13

import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os
import os.path as osp
import collections
from PIL import ImageStat
import torch
import shutil
import attack.config as cfg
from attack.utils.utils import create_dir

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))

def cal_stats(dataset):
    stats = None
    for image, label in dataset:
        if stats is None:
            stats = Stats(image)
        else:
            stats += Stats(image)
    print(f"=> mean: {stats.mean}, std: {stats.stddev}")
    return stats


def categorize_imagenet32(out_name="ILSVRC2012_32x32", train=True):
    data_dir = osp.join(cfg.DATASET_ROOT, "ImageNet32")
    if train:
        out_dir = osp.join(cfg.DATASET_ROOT, out_name, "training_imgs")
        reference_dir = osp.join(cfg.DATASET_ROOT, "ILSVRC2012", "training_imgs")
        split_dir = osp.join(data_dir, "train_32x32")
    else:
        out_dir = osp.join(cfg.DATASET_ROOT, out_name, "val")
        reference_dir = osp.join(cfg.DATASET_ROOT, "ILSVRC2012", "val")
        split_dir = osp.join(data_dir, "valid_32x32")
    meta_dir = osp.join(data_dir, "meta.bin")
    wnid_to_classes, val_wnids = torch.load(meta_dir)
    img2wnid = {}
    count = 0
    for wnid in os.listdir(reference_dir):
        category_dir = osp.join(reference_dir, wnid)
        #print(category_dir)
        for img in os.listdir(category_dir):
            orig_name = img
            if train:
                try:
                    cate, img_id = img.split("_")
                    img_name, _ = img_id.split(".")
                    img_id = img_name + ".png"
                    img_id = "0"*(11-len(img_id)) + img_id
                except:
                    print(img)
            else:
                try:
                    head, _, img_id = img.split("_")
                    img_name, _ = img_id.split(".")
                    if img_name == "50000":
                        continue
                    img_id = img_name + ".png"
                    img_id = img_id[3:]
                    count += 1
                except:
                    print(img)
            img2wnid[(img_id, orig_name)] = wnid
    print(f"found {count} images.")
    
    count = 0
    for (img_id, orig_name), wnid in img2wnid.items():
        cate_dir = osp.join(out_dir, wnid)
        create_dir(cate_dir)
        from_dir = osp.join(split_dir, img_id)
        to_dir = osp.join(cate_dir, orig_name)
        try:
            shutil.copy(from_dir, to_dir)
            count += 1
        except:
            print(f"from_dir: {from_dir}")
            print(f"to_dir: {to_dir}")
            print(f"finished {count}")
    print(f"finished {count}")

if __name__ == "__main__":
    categorize_imagenet32(train=False)