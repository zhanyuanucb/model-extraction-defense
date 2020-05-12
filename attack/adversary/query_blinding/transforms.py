import random
import numpy as np
import kornia
import torch

class RandomTransform:
    def __init__(self, T):
        self.T = T # List of candidate transformations
        
    def __call__(self, x):
        t = random.choice(self.T)
        return t(x)
    
def get_random_gaussian_pt(device="cpu", max_sigma=0.1):
    def add_gaussian_noise(x):
        sigma = np.random.uniform(0, max_sigma)
        gauss = sigma*torch.randn_like(x).to(x.device)
        x_t = torch.clamp(x + gauss, 0., 1.)
        return x_t
    return add_gaussian_noise

def get_random_contrast_pt(device="cpu", min_alpha=0.9, max_alpha=1.4):
    def contrast_random(x):
        #alpha = np.random.uniform(min_alpha, max_alpha)

        alpha = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
        alpha = alpha*(max_alpha - min_alpha) + min_alpha
        x_t = torch.clamp(x*alpha, 0., 1.)

        return x_t
    return contrast_random

def get_random_brightness_pt(device="cpu", min_beta=-0.05, max_beta=0.05):
    def brightness_random(x):
        #beta = np.random.uniform(min_beta, max_beta)
        beta = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
        beta = beta*(max_beta - min_beta) + min_beta
        x_t = torch.clamp(x+beta, 0., 1.)
        return x_t
    return brightness_random

def get_random_rotate_kornia(max_deg=22.5):
    def rotate_random(x):
        rotate = kornia.augmentation.RandomRotation(max_deg, return_transform=False)
        x_t = rotate(x)
        return x_t
    return rotate_random