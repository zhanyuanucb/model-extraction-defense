import os
import os.path as osp
import attack.config as cfg
from attack import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------- Set up trainset/testset
dataset_name = "MNIST"
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

batch_size = 1
num_workers = 10
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

train_count = 1
for image, target in train_loader:
    out_dir = osp.join(cfg.DATASET_ROOT, "mnist/MNIST/processed/train", f"{target[0].data}")
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    image = image[0].squeeze().cpu().numpy()
    plt.imsave(osp.join(out_dir, f"{train_count}.png"), image, cmap='gray')
    train_count += 1

test_count = 1
for image, target in testset:
    out_dir = osp.join(cfg.DATASET_ROOT, "mnist/MNIST/processed/test", f"{target}")
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    image = image[0].squeeze().cpu().numpy()
    plt.imsave(osp.join(out_dir, f"{test_count}.png"), image, cmap='gray')
    test_count += 1