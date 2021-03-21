from __future__ import print_function
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from attack import datasets
from vq_vae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {"batch_size": 256,
          "num_training_updates": 15000,
    
          "num_hiddens": 128,
          "num_residual_hiddens": 32,
          "num_residual_layers": 2,
    
          "embedding_dim": 64,
          "num_embeddings": 512,
    
          "commitment_cost": 0.25,
    
          "decay": 0.99,
    
          "learning_rate": 1e-3,
    
          "log_dir": "./vq-vae_ckpt",
          
          "dataset_name":"CIFAR10"}

batch_size=params['batch_size']
num_training_updates = params["num_training_updates"]

num_hiddens = params["num_hiddens"]
num_residual_hiddens = params["num_residual_hiddens"]
num_residual_layers = params["num_residual_layers"]

embedding_dim = params["embedding_dim"]
num_embeddings = params["num_embeddings"]

commitment_cost = params["commitment_cost"]

decay = params["decay"] 

learning_rate = params["learning_rate"]

log_dir = params["log_dir"]

dataset_name = params["dataset_name"]

valid_datasets = datasets.__dict__.keys()
#if testset_name == "CIFAR10":
#    continue
if dataset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
modelfamily = datasets.dataset_to_modelfamily[dataset_name]
dataset = datasets.__dict__[dataset_name]

test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
try:
    trainset = dataset(train=True, transform=test_transform)
    print('=> Queryset size (training split) = {}'.format(len(trainset)))
    testset = dataset(train=False, transform=test_transform)
    print('=> Queryset size (test split) = {}'.format(len(testset)))
except TypeError as e:
    trainset = dataset(split="train", transform=test_transform) # Augment data while training
    print('=> Queryset size (training split) = {}'.format(len(trainset)))
    testset = dataset(split="valid", transform=test_transform)
    print('=> Queryset size (test split) = {}'.format(len(testset)))



data_variance = np.var(trainset.data / 255.0)

training_loader = DataLoader(trainset, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)


validation_loader = DataLoader(testset,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

vqvae = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = optim.Adam(vqvae.parameters(), lr=learning_rate, amsgrad=False)              

vqvae.train()
train_res_recon_error = []
train_res_perplexity = []
for i in xrange(num_training_updates):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = vqvae(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        torch.save(vqvae.state_dict(), osp.join(log_dir, "./vq.ckpt"))
        print()


train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity_smooth)
ax.set_title('Smoothed Average codebook usage (perplexity).')
ax.set_xlabel('iteration')
plt.savefig(osp.join(log_dir, "loss.png"))

