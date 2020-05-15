import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import attack.utils.model as model_utils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle

class MultiStepJDA:
    def __init__(self, adversary_model, blackbox, mean, std, device, criterion=model_utils.soft_cross_entropy, blinders_fn=None, eps=0.1, steps=1, momentum=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.blinders_fn = blinders_fn
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.momentum = momentum
        self.v = None 
        self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1])
        self.STD = torch.Tensor(std).reshape([1, 3, 1, 1])
        self.device = device
    
    def reset_v(self, input_shape):
        self.v = torch.zeros(input_shape, dtype=torch.float32)#.to(device)

    def get_jacobian(self, images, labels):
        #images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)
        logits = self.adversary_model(images)
        loss = self.criterion(logits, labels.to(torch.long))#.to(logits.device))
        loss.backward()
        jacobian = images.grad.cpu()
        images.requires_grad_(False)
        #conf = F.softmax(logits.detach(), dim=1) 
        return jacobian

    def augment_step(self, images, labels):
        #images, labels = images.to(device), labels.to(device)
        jacobian = self.get_jacobian(images, labels)
        images = images.cpu()
        self.v = self.momentum * self.v + self.lam*torch.sign(jacobian)#.to(device)
        # Clip to valid pixel values
        images = images + self.v
        images = images * self.STD + self.MEAN
        images = torch.clamp(images, 0., 1.)
        images = (images - self.MEAN) / self.STD
        return images

    def augment(self, dataloader):
        """ Multi-step augmentation
        """
        print("Start jocobian data augmentaion...")
        images_aug, labels_aug = [], []
        for images, labels in dataloader:
            self.reset_v(input_shape=images.shape)
            images, labels = images.to(self.device), labels.to(self.device)
            if self.blinders_fn is not None:
                images = self.blinders_fn(images)
            for i in range(self.steps):
                images = Variable(images, requires_grad=True)
                images = self.augment_step(images, labels)
                images = images.to(self.device)
            images = images.cpu()
            is_adv, y = self.blackbox(images)  # Inspection
            images_aug.append(images.clone())
            labels_aug.append(y.cpu().clone())
        return torch.cat(images_aug), torch.cat(labels_aug)
    
    def __call__(self, dataloader):
        images_aug, labels_aug = self.augment(dataloader) 
        return images_aug, labels_aug