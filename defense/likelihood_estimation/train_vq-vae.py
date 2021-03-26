from __future__ import print_function
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp
import json
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from attack import datasets
from attack.utils.utils import create_dir
from vq_vae import VQVAE



def main():
    parser = argparse.ArgumentParser(description='Simulate benign users')
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=32)
    parser.add_argument("--num_training_updates", metavar="TYPE", type=int, default=15000)
    parser.add_argument("--num_hiddens", metavar="TYPE", type=int, default=128)
    parser.add_argument("--num_residual_hiddens", metavar="TYPE", type=int, default=32)
    parser.add_argument("--num_residual_layers", metavar="TYPE", type=int, default=2)
    parser.add_argument("--embedding_dim", metavar="TYPE", type=int, default=64)
    parser.add_argument("--num_embeddings", metavar="TYPE", type=int, default=512)
    parser.add_argument("--commitment_cost", metavar="TYPE", type=float, help="binary search lowerbound", default=0.25)
    parser.add_argument("--decay", metavar="TYPE", type=float, help="binary search lowerbound", default=0.99)
    parser.add_argument("--learning_rate", metavar="TYPE", type=float, help="binary search lowerbound", default=1e-3)
    parser.add_argument("--log_dir", metavar="PATH", type=str, default="./vq-vae_ckpt")
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="")
    parser.add_argument("--dataset_name", metavar="TYPE", type=str, default="CIFAR10")
    parser.add_argument("--div_by_var", action="store_true")
    parser.add_argument("--stream_var", action="store_true")
    parser.add_argument('--train_on_seed', action='store_true')
    parser.add_argument('--seedsize', metavar='TYPE', type=int, help='size of seed images', default=5000)
    parser.add_argument("--device_id", metavar="TYPE", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #params = {"batch_size": 256,
    #          "num_training_updates": 15000,
    #    
    #          "num_hiddens": 128,
    #          "num_residual_hiddens": 32,
    #          "num_residual_layers": 2,
    #    
    #          "embedding_dim": 64,
    #          "num_embeddings": 512,
    #    
    #          "commitment_cost": 0.25,
    #    
    #          "decay": 0.99,
    #    
    #          "learning_rate": 1e-3,
    #    
    #          "log_dir": "./vq-vae_ckpt",
    #          
    #          "dataset_name":"CIFAR10"}

    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19]

    created_on = params['created_on']

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

    log_dir = osp.join(params["log_dir"], created_on+'-'+params['log_suffix'])
    create_dir(log_dir)

    dataset_name = params["dataset_name"]

    valid_datasets = datasets.__dict__.keys()
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

    if params["div_by_var"]:
        data_variance = np.var(trainset.data / 255.0)
    else:
        data_variance = 1.

    if params['stream_var']:
        num_samples = 0
        pixel_sum = 0.
        pixel_sqr_sum = 0.

    ##########################################
    # Using seed images
    ##########################################
    if params['train_on_seed']:
        trainset_full = trainset
        seed_idx = np.random.choice(range(len(trainset)), size=params['seedsize'], replace=False)
        train_idx, _ = train_test_split(seed_idx, test_size=0.1, random_state=42)
        trainset = Subset(trainset_full, train_idx)
        #valset = Subset(trainset_full, val_idx)

    training_loader = DataLoader(trainset, 
                                 batch_size=batch_size, 
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
        B, C, H, W = data.shape

        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = vqvae(data)

        if params['stream_var']:
            num_samples += B*C*H*W
            # Calculate data variance from stream data
            pixel_sum += torch.sum(data).item()
            pixel_sqr_sum += torch.sum(data**2).item()
            data_variance = (pixel_sqr_sum - pixel_sum**2/num_samples)/num_samples

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

    params_out_path = osp.join(log_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()