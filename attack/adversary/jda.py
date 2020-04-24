import torch
import torch.nn as nn
import torch.nn.functional as F
import attack.utils.model as model_utils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
class MultiStepJDA:
    def __init__(self, adversary_model, blackbox, MEAN, STD, criterion=model_utils.soft_cross_entropy, eps=0.1, steps=1, momentum=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.momentum = momentum
        self.v = None 
        self.MEAN = torch.Tensor(MEAN).reshape([1, 3, 1, 1])
        self.STD = torch.Tensor(STD).reshape([1, 3, 1, 1])
    
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
        conf = F.softmax(logits.detach(), dim=1) 
        return jacobian, conf # Inspection

    def augment_step(self, images, labels):
        #images, labels = images.to(device), labels.to(device)
        jacobian, conf = self.get_jacobian(images, labels)
        images = images.cpu()
        self.v = self.momentum * self.v + self.lam*torch.sign(jacobian)#.to(device)
        # Clip to valid pixel values
        images = images + self.v
        images = images * self.STD + self.MEAN
        images = torch.clamp(images, 0., 1.)
        images = (images - self.MEAN) / self.STD
        return images, conf.cpu().numpy() # Inspection

    def augment(self, dataloader):
        """ Multi-step augmentation
        """
        print("Start jocobian data augmentaion...")
        images_aug, labels_aug = [], []
        is_advs, confs = [], [] # Inspection
        for images, labels in dataloader:
            self.reset_v(input_shape=images.shape)

            for _ in range(self.steps):
                images, labels = images.to(device), labels.to(device)
                images, conf = self.augment_step(images, labels)
                is_adv, y = self.blackbox(images)  # Inspection
                images_aug.append(images.clone())
                labels_aug.append(y.cpu().clone())
                confs.append(conf) # TODO: Select by is_adv
                is_advs.append(is_adv)
        return torch.cat(images_aug), torch.cat(labels_aug), np.concatenate(is_advs), np.concatenate(confs)
    
    def __call__(self, dataloader):
        return self.augment(dataloader)