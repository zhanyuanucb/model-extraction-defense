import torch
import torch.nn as nn
import attack.utils.model as model_utils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
MEAN = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
STD = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
class MultiStepJDA:
    def __init__(self, adversary_model, blackbox, criterion=model_utils.soft_cross_entropy, eps=0.1, batchsize=64, input_shape=[3, 224, 224], steps=1, momentum=0):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.momentum = momentum
        self.v = None 
    
    def reset_v(self, batchsize, input_shape=[3, 224, 224]):
        self.v = torch.zeros([batchsize]+input_shape, dtype=torch.float32).to(device)

    def get_jacobian(self, images, labels):
        images.requires_grad_(True)
        logits = self.adversary_model(images)
        loss = self.criterion(logits, labels.to(torch.long).to(logits.device))
        loss.backward()
        jacobian = images.grad.cpu()
        images.requires_grad_(False)
        return jacobian

    def augment_step(self, images, labels):
        images, labels = images.to(device), labels.to(device)
        jacobian = self.get_jacobian(images, labels)
        self.v = self.momentum * self.v + self.lam*torch.sign(jacobian).to(device)
        return images + self.v

    def augment(self, dataloader, out_dir):
        """ Multi-step augmentation
        """
        print("Start jocobian data augmentaion...")
        augset = []
        img_count = 0
        for images, labels in dataloader:
            batchsize = images.shape[0]
            self.reset_v(batchsize)

            for _ in range(self.steps):
                images = self.augment_step(images, labels)
                images = images.to(device)
                y = self.blackbox(images)
                for i in range(images.size(0)):
                    image_i = images[i].squeeze().cpu().numpy()
                    image_i = image_i*STD + MEAN # undo preprocessing
                    image_i = image_i.transpose([1, 2, 0])
                    # TODO: Confirm the range of valid pixel value
                    image_i = np.clip(image_i, 0, 1)
                    save_path_i = osp.join(out_dir, f"{img_count}.png")
                    plt.imsave(save_path_i, image_i)
                    img_count += 1
                    augset.append((save_path_i, y[i].cpu().squeeze()))
        return augset
    
    def __call__(self, dataloader, out_dir):
        return self.augment(dataloader, out_dir)