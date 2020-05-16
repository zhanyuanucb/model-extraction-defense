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

def get_uniform_noise(device="cpu", r=0.064):

   def add_uniform_noise(x):
       noise = 2*r*torch.rand_like(x).to(x.device) - r
       x_t = torch.clamp(x + noise, 0., 1.)
       return x_t

   return add_uniform_noise

def get_gaussian_noise(device="cpu", sigma=0.095):

   def add_gaussian_noise(x):
       noise = sigma*torch.randn_like(x).to(x.device)
       x_t = torch.clamp(x + noise, 0., 1.)
       return x_t

   return add_gaussian_noise

def get_random_contrast(device="cpu", min_alpha=-0.55, max_alpha=0.55):
    def contrast_random(x):
        #alpha = np.random.uniform(min_alpha, max_alpha)

        alpha = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
        alpha = alpha*(max_alpha - min_alpha) + min_alpha
        x_t = torch.clamp(x*alpha, 0., 1.)

        return x_t
    return contrast_random

def get_random_brightness(device="cpu", min_beta=-0.09, max_beta=0.09):
    def brightness_random(x):
        #beta = np.random.uniform(min_beta, max_beta)
        beta = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
        beta = beta*(max_beta - min_beta) + min_beta
        x_t = torch.clamp(x+beta, 0., 1.)
        return x_t
    return brightness_random

def get_random_rotate(device="cpu", max_deg=0.018):
    def rotate_random(x):
        rotate = kornia.augmentation.RandomRotation(max_deg, return_transform=False)
        x_t = rotate(x)
        return x_t
    return rotate_random

def get_random_translate(device="cpu", r=0.45):
    def translate_random(x):
        translate = kornia.augmentation.RandomAffine(0, translate=(r, r), return_transform=False)
        x_t = translate(x)
        return x_t
    return translate_random

def get_random_scale(device="cpu", r=0.17):
    def scale_random(x):
        scale = kornia.augmentation.RandomAffine(0, scale=(1-r, 1+r), return_transform=False)
        x_t = scale(x)
        return x_t
    return scale_random

def get_random_crop(device="cpu", r=0.04):
    def crop_random(x):
        crop = kornia.augmentation.RandomResizedCrop(size=(32, 32), scale=(1-r, 1.), ratio=(1., 1.), return_transform=False)
        x_t = crop(x)
        return x_t
    return crop_random