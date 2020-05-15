import argparse
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
sys.path.append('/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding')
import os.path as osp
import pickle
from datetime import datetime
import torch
import torchvision.models as models

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
import modelzoo.zoo as zoo
from detector import *
from attack.adversary.jda import MultiStepJDA
from attack.adversary.query_blinding.blinders import AutoencoderBlinders
import attack.adversary.query_blinding.transforms as blinders_transforms
from utils import ImageTensorSet

__author = "Zhanyuan Zhang"
__author_email__ = "zhang_zhanyuan@berkeley.edu"
__reference__ = "https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/train.py"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Simulate benign users')
    parser.add_argument("--num_classes", metavar="TYPE", type=int, default=10)
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=32)
    parser.add_argument("--blackbox_dir", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28")
    parser.add_argument("--testset_name", metavar="TYPE", type=str, default="CIFAR10")
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--encoder_ckp", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--k", metavar="TYPE", type=int, default=10)
    parser.add_argument("--thresh", metavar="TYPE", type=float, help="detector threshold", default=0.0397684188708663)
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="benign")
    parser.add_argument("--log_dir", metavar="PATH", type=str,
                        default="./")
    args = parser.parse_args()
    params = vars(args)

    # ------------ Start
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    if use_cuda:
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print(device)
    gpu_count = torch.cuda.device_count()

    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19]
    created_on = params['created_on']
    # ----------- Initialize Detector
    k = params["k"]
    thresh = params["thresh"]
    log_suffix = params["log_suffix"]
    log_dir = params["log_dir"]
    testset_name = params["testset_name"]
    encoder_arch_name = params["encoder_arch_name"]
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    num_classes = 10
    encoder = zoo.get_net(encoder_arch_name, modelfamily, num_classes=num_classes)
    MEAN, STD = cfg.NORMAL_PARAMS[modelfamily]

    # ----------- Setup Similarity Encoder
    blackbox_dir = params["blackbox_dir"]
    encoder_ckp = params["encoder_ckp"]
    if encoder_ckp is not None:
        encoder_margin = params["encoder_margin"]
        encoder_ckp = osp.join(encoder_ckp, encoder_arch_name, f"{testset_name}-margin-{encoder_margin}")
        ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
        print(f"=> Loading similarity encoder checkpoint '{ckp}'")
        checkpoint = torch.load(ckp)
        start_epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        encoder = encoder.to(device)

        blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on)
    else:
        blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Set up query set
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    batch_size = params["batch_size"]
    num_workers = 10

    query_train_dir = cfg.dataset2dir[testset_name]["train"]
    query_train_images, query_train_labels = torch.load(query_train_dir)
    query_train_images = query_train_images.permute(0, 3, 1, 2)
    query_train_samples = (query_train_images, query_train_labels)
    query_trainset = ImageTensorSet(query_train_samples, transform=test_transform)
    queryloader_train = DataLoader(query_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('=> Queryset size (training split) = {}'.format(len(query_trainset)))
    query_val_dir = cfg.dataset2dir[testset_name]["test"]
    query_val_images, query_val_labels = torch.load(query_val_dir)
    query_val_images = query_val_images.permute(0, 3, 1, 2)
    query_val_samples = (query_val_images, query_val_labels)
    query_valset = ImageTensorSet(query_val_samples, transform=test_transform)
    queryloader_val = DataLoader(query_valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('=> Queryset size (validation split) = {}'.format(len(query_valset)))

    #--------- Extraction
    for images, labels in queryloader_train:
        is_adv, y = blackbox(images)

    for images, labels in queryloader_val:
        is_adv, y = blackbox(images)

if __name__ == '__main__':
    main()