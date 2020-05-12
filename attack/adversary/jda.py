import torch
import torch.nn as nn
import torch.nn.functional as F
import attack.utils.model as model_utils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle

class MultiStepJDA:
    def __init__(self, adversary_model, blackbox, mean, std, device, criterion=model_utils.soft_cross_entropy, eps=0.1, steps=1, momentum=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
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
            images, labels = images.to(self.device), labels.to(self.device)

            for _ in range(self.steps):
                images, conf = self.augment_step(images, labels)
                is_adv, y = self.blackbox(images)  # Inspection
                images_aug.append(images.clone())
                labels_aug.append(y.cpu().clone())
                confs.append(conf) 
                is_advs.append(is_adv)
        return torch.cat(images_aug), torch.cat(labels_aug), np.concatenate(is_advs), np.concatenate(confs)
    
    def __call__(self, dataloader):
        images_aug, labels_aug, is_advs, confs = self.augment(dataloader) 
        adv_confs_batch = [confs[i] for i in range(confs.shape[0]) if is_advs[i]]
        batch_size = images_aug.size(0)

        # Filter by confidence
        #cond = [max(conf) <= 1. for conf in confs] # if don't apply filtering
        cond = [max(conf) < 0.9 for conf in confs] 

        # Randomly pick fraction of k samples
        #k = 0.6
        #indices = np.random.choice(batch_size, round(batch_size*k), replace=False)
        #cond = [False for _ in range(batch_size)]
        #for idx in indices:
        #    cond[idx] = True

        cleaned_images = torch.stack([images_aug[i] for i in range(batch_size) if cond[i]])
        cleaned_labels = torch.stack([labels_aug[i] for i in range(batch_size) if cond[i]])
        cleaned_is_advs = [is_advs[i] for i in range(batch_size) if cond[i]]
        cleaned_confs = np.array([confs[i] for i in range(batch_size) if cond[i]])
        return cleaned_images, cleaned_labels, adv_confs_batch