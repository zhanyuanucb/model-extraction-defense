import random
import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, VisionDataset, IMG_EXTENSIONS, default_loader
from torchvision.transforms import transforms

from PIL import Image
#import foolbox
#from foolbox.attacks import LinfPGD as PGD
#from foolbox.attacks import L2CarliniWagnerAttack
#from foolbox.criteria import Misclassification, TargetedMisclassification

class ImageTensorSet(Dataset):
    """
    Data are saved as:
    List[data:torch.Tensor(), labels:torch.Tensor()]
    """
    def __init__(self, samples, transform=None, dataset="cifar"):
        self.data, self.targets = samples
        self.transform = transform
        self.mode = "RGB" if dataset != "mnist" else "L"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img *= 255
            img = img.numpy().astype('uint8')
            if self.mode == "RGB":
                img = img.transpose([1, 2, 0])
            img = Image.fromarray(img, mode=self.mode)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class ImageTensorSetKornia(Dataset):
    """
    Data are saved as:
    List[data:torch.Tensor(), labels:torch.Tensor()]
    """
    def __init__(self, samples, transform=None, dataset="cifar"):
        self.data, self.targets = samples
        self.transform = transform
        self.mode = "RGB" if dataset != "mnist" else "L"

    def __getitem__(self, index):
        img, target = self.data[index][None], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)[0]

        return img, target

    def __len__(self):
        return len(self.data)


class PositiveNegativeSet(VisionDataset):
    """
    For data in form of serialized tensor
    """
    def __init__(self, load_path, normal_transform=None, random_transform=None, dataset="CIFAR10"):
        self.data, self.targets = torch.load(load_path)
        self.n_samples = self.data.size(0)
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.mode = "L" if dataset == "MNIST" else "RGB"

    def __getitem__(self, index):
        img_pt, y = self.data[index], self.targets[index]

        if not isinstance(img_pt[0][0][0], np.uint8):
            img_pt *= 255
        if not isinstance(img_pt, np.ndarray):
            img_pt = img_pt.numpy()
        img = Image.fromarray(img_pt.astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        ori_img = self.normal_transform(img)
        ran_img = self.random_transform(img)

        other_idx = random.choice(list(range(index)) + list(range(index+1, self.n_samples)))
        img2_pt = self.data[other_idx].clone()
        #img2_pt = self.data[other_idx].copy()
        if not isinstance(img2_pt[0][0][0], np.uint8):
            img2_pt *= 255
        if not isinstance(img2_pt, np.ndarray):
            img2_pt = img2_pt.numpy()
        img2 = Image.fromarray(img2_pt.astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        other_img = self.normal_transform(img2)

        return ori_img, ran_img, other_img, y

    def __len__(self):
        return self.n_samples

class BinarySampleSet(PositiveNegativeSet):
    """
    For data in form of serialized tensor
    """
    def __init__(self, load_path, normal_transform=None, random_transform=None, dataset="CIFAR10"):
        super(BinarySampleSet, self).__init__(load_path=load_path, normal_transform=normal_transform, random_transform=random_transform, dataset=dataset)
        self.data, self.targets = torch.load(load_path)
        self.n_samples = self.data.size(0)
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.mode = "L" if dataset == "MNIST" else "RGB"
        #self.attack = attack
        #self.fmodel = fmodel
        #self.eps = eps
        #self.adv_criterion = adv_criterion
        #self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1])
        #self.STD = torch.Tensor(std).reshape([1, 3, 1, 1])
        print("=> Initialize BinarySampleSet")

    #def normalize(self, images):
    #    mean, std = self.MEAN.to(images.device), self.STD.to(images.device)
    #    return (images-mean) / std

    def __getitem__(self, index):
        img_pt, y = self.data[index], self.targets[index]

        if not isinstance(img_pt[0][0][0], np.uint8):
            img_pt *= 255
        if not isinstance(img_pt, np.ndarray):
            img_pt = img_pt.numpy()

        img = Image.fromarray(img_pt.astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        ori_img = self.normal_transform(img)

        other_idx = random.choice(list(range(index)) + list(range(index+1, self.n_samples)))
        if isinstance(self.data, np.ndarray):
            img2_pt = self.data[other_idx].copy()
        else:
            img2_pt = self.data[other_idx].clone()
        if not isinstance(img2_pt[0][0][0], np.uint8):
            img2_pt *= 255
        if not isinstance(img2_pt, np.ndarray):
            img2_pt = img2_pt.numpy()
        img2 = Image.fromarray(img2_pt.astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        other_img = self.normal_transform(img2)

        return ori_img, other_img, y

    def __len__(self):
        return self.n_samples

class PositiveNegativeImageSet(ImageFolder):
    """
    For data in form of serialized tensor
    """
    def __init__(self, root, normal_transform=None, random_transform=None):
        super(PositiveNegativeImageSet, self).__init__(root)
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.n_samples = len(self.imgs)

    def __getitem__(self, index):
        img_path, y = self.imgs[index]
        img = self.loader(img_path)

        ori_img = self.normal_transform(img)
        ran_img = self.random_transform(img)

        other_idx = random.choice(list(range(index)) + list(range(index+1, self.n_samples)))
        img2_path, _ = self.imgs[other_idx]
        img2 = self.loader(img2_path)
        other_img = self.normal_transform(img2)

        return ori_img, ran_img, other_img, y


class BlinderPositiveNegativeSet(VisionDataset):
    """
    For data in form of serialized tensor
    """
    def __init__(self, load_path, encoder, device="cpu", normal_transform=None, random_transform=None, dataset="CIFAR10"):
        self.data, self.targets = torch.load(load_path)
        self.n_samples = self.data.size(0)
        self.normal_transform = normal_transform
        self.random_transform = random_transform
        self.mode = "L" if dataset == "MNIST" else "RGB"
        self.encoder = encoder
        self.device = device

    def __getitem__(self, index):
        img_pt, y = self.data[index], self.targets[index]

        img = img_pt*255
        img = Image.fromarray(img.numpy().astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        ori_img = self.normal_transform(img)

        with torch.no_grad():
            img_blinder = torch.clamp(self.encoder(img_pt[None].to(self.device)), 0., 1.)
        img_blinder = img_blinder[0]*255
        img_blinder = Image.fromarray(img_blinder.cpu().numpy().astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        img_blinder = self.random_transform(img_blinder)

        other_idx = random.choice(list(range(index)) + list(range(index+1, self.n_samples)))
        img2_pt = self.data[other_idx].clone()
        img2_pt *= 255
        img2 = Image.fromarray(img2_pt.numpy().astype('uint8').transpose([1, 2, 0]), mode=self.mode)
        other_img = self.normal_transform(img2)

        return ori_img, img_blinder, other_img, y

    def __len__(self):
        return self.n_samples

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

class MNISTSeedsetImagePaths(ImageFolder):
    """MNIST Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image = image[0][None] # only use the first channel
        return image, target

class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class IdLayer(nn.Module):
    def __init__(self, activation=None):
        super(IdLayer, self).__init__()
        self.activation = activation
    def forward(self, x):
        if self.activation is not None:
            return self.activation(x)
        return x