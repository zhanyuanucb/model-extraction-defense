import random
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
                 criterion=model_utils.soft_cross_entropy, t_rand=False,
                 blinders_fn=None, eps=0.1, steps=1, momentum=0, delta_step=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.blinders_fn = blinders_fn
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.delta_step = delta_step
        self.t_rand = t_rand
        self.momentum = momentum
        self.v = None 
        self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1]).to(device)
        self.STD = torch.Tensor(std).reshape([1, 3, 1, 1]).to(device)
        self.device = device
    
    def get_seedset(self, dataloader):
        images, labels = [], []
        for x_t, _ in dataloader:
            x_t = x_t.cuda()
            is_adv, y_t = self.blackbox(x_t)
            y_t = y_t.cpu()
            images.append(x_t.cpu())
            labels.append(y_t)

        seedset = [torch.cat(images), torch.cat(labels)]
        return seedset

    def reset_v(self, input_shape):
        self.v = torch.zeros(input_shape, dtype=torch.float32).to(self.device)

    def get_jacobian(self, images, labels):
        #images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)
        logits = self.adversary_model(images)
        loss = self.criterion(logits, labels.to(torch.long))#.to(logits.device))
        loss.backward()
        jacobian = images.grad
        images.requires_grad_(False)
        #conf = F.softmax(logits.detach(), dim=1) 
        return jacobian

    def augment_step(self, images, labels):
        #images, labels = images.to(device), labels.to(device)
        jacobian = self.get_jacobian(images, labels)
        if self.t_rand:
            jacobian *= -1
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
        conf_list = []
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            if self.t_rand:
                #TODO: implement T-RAND
                targeted_labels = torch.zeros_like(labels)
                batch_size, num_class = labels.size(0), labels.size(1)
                _, cur_labels = torch.topk(labels, 1)
                indices = [[] for _ in range(batch_size)]
                for i in range(batch_size):
                    indices[i].append(random.choice([j for j in range(num_class) if j != cur_labels[i]])) 

                indices = torch.Tensor(indices).to(torch.long)
                targeted_labels.scatter(1, indices.to(self.device), torch.ones_like(labels))

                labels = targeted_labels.to(self.device)
            self.reset_v(input_shape=images.shape)

#            if self.blinders_fn is not None:
#                images = images * self.STD + self.MEAN
#                images = torch.clamp(images, 0., 1.)
#                images = self.blinders_fn(images)
#                images = (images - self.MEAN) / self.STD

            for i in range(self.steps):
                images = Variable(images, requires_grad=True)
                images = self.augment_step(images, labels)

            images_aug.append(images.cpu())

            if self.blinders_fn is not None:
                with torch.no_grad():
                    images = images * self.STD + self.MEAN
                    images = torch.clamp(images, 0., 1.)
                    images = self.blinders_fn(images)
                    images = (images - self.MEAN) / self.STD

            is_adv, y = self.blackbox(images.cpu())  # Inspection
#            y = self.blackbox(images)  # Inspection
            labels_aug.append(y.cpu())
        self.steps += self.delta_step
        return torch.cat(images_aug), torch.cat(labels_aug)
    
    def __call__(self, dataloader):
        return self.augment(dataloader)