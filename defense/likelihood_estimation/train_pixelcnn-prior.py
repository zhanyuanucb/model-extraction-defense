"""Main training script for models."""
import os
import argparse
from datetime import datetime
import json
import sys
import os.path as osp
sys.path.append('./pytorch-generative')
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from six.moves import xrange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.distributions.categorical import Categorical

import pytorch_generative as pg
from vq_vae import VQVAE
from attack import datasets
from attack.utils.utils import create_dir

MODEL_DICT = {
    "gated_pixel_cnn": pg.models.gated_pixel_cnn,
    "image_gpt": pg.models.image_gpt,
    "made": pg.models.made,
    "nade": pg.models.nade,
    "pixel_cnn": pg.models.pixel_cnn,
    "pixel_snail": pg.models.pixel_snail,
    "vae": pg.models.vae,
    "beta_vae": pg.models.beta_vae,
    "vd_vae": pg.models.vd_vae,
    "vq_vae": pg.models.vq_vae,
    "vq_vae_2": pg.models.vq_vae_2,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_updates", type=int, help="number of training epochs", default=30000)
    parser.add_argument("--lr", metavar="TYPE", type=float, help="binary search lowerbound", default=1e-3)
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=100)
    parser.add_argument("--num_hiddens", metavar="TYPE", type=int, default=128)
    parser.add_argument("--num_residual_hiddens", metavar="TYPE", type=int, default=32)
    parser.add_argument("--num_residual_layers", metavar="TYPE", type=int, default=2)
    parser.add_argument("--embedding_dim", metavar="TYPE", type=int, default=64)
    parser.add_argument("--num_embeddings", metavar="TYPE", type=int, default=512)
    parser.add_argument("--commitment_cost", metavar="TYPE", type=float, help="binary search lowerbound", default=0.25)
    parser.add_argument("--decay", metavar="TYPE", type=float, help="binary search lowerbound", default=0.99)
    parser.add_argument("--vqvae_ckpt", metavar="PATH", type=str, default=None)
    parser.add_argument("--log_dir", metavar="PATH", type=str, default="./vq-vae_ckpt")
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="")
    parser.add_argument('--train_on_seed', action='store_true')
    parser.add_argument('--seedsize', metavar='TYPE', type=int, help='size of seed images', default=5000)
    parser.add_argument("--device_id", metavar="TYPE", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ###################################
    # Prepare VQ-VAE
    ###################################
    num_hiddens = params["num_hiddens"]
    num_residual_hiddens = params["num_residual_hiddens"]
    num_residual_layers = params["num_residual_layers"]
    embedding_dim = params["embedding_dim"]
    num_embeddings = params["num_embeddings"]
    commitment_cost = params["commitment_cost"]
    vqvae_ckpt = params["vavqe_ckpt"]

    decay = params["decay"] 
    vqvae = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, 
                  commitment_cost, decay).to(device)
    vqvae.load_ckpt(vqvae_ckpt)
    vqvae.eval()

    ###################################
    # Prepare train and validation sets
    ###################################
    dataset_name = params["dataset_name"]
    batch_size = params["batch_size"]

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

    if params['train_on_seed']:
        trainset_full = trainset
        seed_idx = np.random.choice(range(len(trainset)), size=params['seedsize'], replace=False)
        train_idx, val_idx = train_test_split(seed_idx, test_size=0.1, random_state=42)
        trainset = Subset(trainset_full, train_idx)
        valset = Subset(trainset_full, val_idx)

    training_loader = DataLoader(trainset, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    eval_loader = DataLoader(valset, 
                             batch_size=batch_size, 
                             shuffle=True)


    ##########################################
    # Set up for PixcelCNN prior training
    ##########################################
    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19]
    created_on = params['created_on']
    log_dir = osp.join(params["log_dir"], created_on+'-'+params['log_suffix'])
    learning_rate = params["lr"]
    model = MODEL_DICT['gated_pixel_cnn']

    #code_size = 32
    in_channels= 1
    out_channels = num_embeddings
    n_gated = 11 #num_layers_pixelcnn = 12
    gated_channels = 32 #fmaps_pixelcnn = 32
    head_channels = 32

    pixel_cnn = model.GatedPixelCNN(in_channels=in_channels, 
                                    out_channels=out_channels,
                                    n_gated=n_gated,
                                    gated_channels=gated_channels, 
                                    head_channels=head_channels)

    pixel_cnn = pixel_cnn.to(device)
    optimizer = optim.Adam(pixel_cnn.parameters(), lr=learning_rate, amsgrad=False)              


    ###########################################
    # Start training
    ###########################################
    num_training_updates = params['num_training_updates']
    num_iter_per_epoch = num_training_updates//len(trainset)
    grad_clip = 1.
    loss_fn = nn.CrossEntropyLoss()
    val_loss_hist = []
    best_val_loss = float("inf")
    training_loader_iter = iter(training_loader)
    for i in xrange(num_training_updates):
        pixel_cnn.train()
        (input, _) = next(training_loader_iter)

        input = input.to(device)
        optimizer.zero_grad()

        #vq_loss, data_recon, perplexity = vqvae(data)
        vq_output = vqvae._pre_vq_conv(vqvae._encoder(input))
        _, z, _, _ = vqvae._vq_vae(vq_output)

        logits = pixel_cnn(z)
        logits = logits.permute(0, 2, 3, 1).view(-1, num_embeddings)
        #px = Categorical(logits=logits)
        #sampled_pixelcnn = px.sample()
        #log_prob = px.log_prob(sampled_pixelcnn)
        input = input.permute(0, 2, 3, 1) # BCHW -> BHWC
        label = input.view(-1,).to(torch.long)
        loss = loss_fn(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm(pixel_cnn.parameters(), grad_clip)
        optimizer.step()

        if (i+1) % num_iter_per_epoch == 0:
            pixel_cnn.eval()
            val_loss = 0.
            total = 0
            with torch.no_grad():
                for i, (input, _) in enumerate(eval_loader):
                    input = input.to(device)
                    total += input.size(0)

                    vq_output = vqvae._pre_vq_conv(vqvae._encoder(input))
                    _, z, _, _ = vqvae._vq_vae(vq_output)

                    logits = pixel_cnn(z)
                    logits = logits.permute(0, 2, 3, 1).view(-1, num_embeddings)
                    #px = Categorical(logits=logits)
                    #sampled_pixelcnn = px.sample()
                    #log_prob = px.log_prob(sampled_pixelcnn)
                    input = input.permute(0, 2, 3, 1) # BCHW -> BHWC
                    label = input.view(-1,).to(torch.long)
                    loss = loss_fn(logits, label)
                    val_loss += loss.item()
            val_loss /= total
            val_loss_hist.append(val_loss)
            if val_loss < best_val_loss:
                print(f"Val Loss at epoch {(i+1) // num_iter_per_epoch}: {val_loss}")
                torch.save(pixel_cnn.state_dict(), osp.join(log_dir, "./pixelcnn.ckpt"))
                print()

    plt.plot()
    plt.plot(val_loss_hist)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.savefig(osp.join(log_dir, "val_loss_hist.png"))
    plt.show()

    params_out_path = osp.join(log_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == "__main__":
    main()
