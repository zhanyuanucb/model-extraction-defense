import os
import os.path as osp
import attack.config as cfg
from attack import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------- Set up trainset/testset
dataset_name = "CIFAR10"
num_classes = 10
valid_datasets = datasets.__dict__.keys()
modelfamily = datasets.dataset_to_modelfamily[dataset_name]
if dataset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
dataset = datasets.__dict__[dataset_name]
trainset = dataset(train=True, transform=transforms.ToTensor())
testset = dataset(train=False, transform=transforms.ToTensor())
if len(testset.classes) != num_classes:
    raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

batch_size = 128
num_workers = 10
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

images_train, labels_train = [], []
for images, targets in train_loader:
    out_dir = osp.join(cfg.DATASET_ROOT, "cifar10/training.pt")
    images_train.append(images)
    labels_train.append(targets)
    train_pt = [torch.cat(images_train).permute(0, 2, 3, 1), torch.cat(labels_train)]
torch.save(train_pt, out_dir)
print(f"Save {train_pt[0].size(0)} images to {out_dir}")

images_test, labels_test = [], []
for images, targets in test_loader:
    out_dir = osp.join(cfg.DATASET_ROOT, "cifar10/test.pt")
    images_test.append(images)
    labels_test.append(targets)
    test_pt = [torch.cat(images_test).permute(0, 2, 3, 1), torch.cat(labels_test)]
torch.save(test_pt, out_dir)
print(f"Save {test_pt[0].size(0)} images to {out_dir}")