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
    def __init__(self, adversary_model, blackbox, mean, std, device, 
                 criterion=model_utils.soft_cross_entropy, return_conf_max=False,
                 blinders_fn=None, eps=0.1, steps=1, momentum=0, delta_step=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.blinders_fn = blinders_fn
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.delta_step = delta_step
        self.momentum = momentum
        self.v = None 
        self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1])
        self.STD = torch.Tensor(std).reshape([1, 3, 1, 1])
        self.device = device
        self.return_conf_max = return_conf_max
    
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
        print(f"JDA steps: {self.steps}")
        images_aug, labels_aug = [], []
        conf_list = []
        for images, labels in dataloader:
            self.reset_v(input_shape=images.shape)
            if self.blinders_fn is not None:
                images = images * self.STD + self.MEAN
                images = torch.clamp(images, 0., 1.)
                images = self.blinders_fn(images)
                images = (images - self.MEAN) / self.STD
            images, labels = images.to(self.device), labels.to(self.device)
            for i in range(self.steps):
                images = Variable(images, requires_grad=True)
                images = self.augment_step(images, labels)
                images = images.to(self.device)
            images = images.cpu()
            images_aug.append(images.clone())
            if self.return_conf_max:
                is_adv, y, conf_max = self.blackbox(images)  # Inspection
                conf_list.append(conf_max.clone())
            else:
                is_adv, y = self.blackbox(images)  # Inspection
                #y = self.blackbox(images)  # Inspection
            labels_aug.append(y.cpu().clone())
        self.steps += self.delta_step
        if self.return_conf_max:
            conf_list = torch.cat(conf_list).cpu()
            return torch.cat(images_aug), torch.cat(labels_aug), conf_list
        return torch.cat(images_aug), torch.cat(labels_aug)
    
    def __call__(self, dataloader):
        return self.augment(dataloader)