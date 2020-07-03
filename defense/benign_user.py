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
from attack.utils.utils import create_dir
from attack import datasets
import modelzoo.zoo as zoo
from detector import *
from attack.adversary.jda import MultiStepJDA
from attack.adversary.query_blinding.blinders import AutoencoderBlinders
import attack.adversary.query_blinding.transforms as blinders_transforms
from utils import ImageTensorSet, IdLayer

__author__ = "Zhanyuan Zhang"
__author_email__ = "zhang_zhanyuan@berkeley.edu"
__reference__ = "https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/train.py"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Simulate benign users')
    parser.add_argument("--num_classes", metavar="TYPE", type=int, default=10)
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=32)
    parser.add_argument("--blackbox_dir", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28_2")
    parser.add_argument("-l", "--testset_names", nargs='+', type=str, required=True)
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--encoder_ckp", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--encoder_suffix", metavar="TYPE", type=str, default="")
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument("--k", metavar="TYPE", type=int, default=1)
    parser.add_argument("--thresh", metavar="TYPE", type=float, help="detector threshold", default=0.0012760052197845653)
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="benign")
    parser.add_argument("--log_dir", metavar="PATH", type=str,
                        default="./benign_log")
    parser.add_argument("--device_id", metavar="TYPE", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)

    # ------------ Start
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19]
    created_on = params['created_on']
    # ----------- Initialize Detector
    k = params["k"]
    thresh = params["thresh"]
    log_suffix = params["log_suffix"]
    log_dir = params["log_dir"]
    created_on = str(datetime.now()).replace(' ', '_')[:19]
    log_dir = osp.join(log_dir, created_on)
    create_dir(log_dir)

    encoder_arch_name = params["encoder_arch_name"]
    num_classes = 10
    encoder = zoo.get_net(encoder_arch_name, "cifar", num_classes=num_classes)
    activation_name = params['activation']
    if activation_name == "sigmoid":
        activation = nn.Sigmoid()
        print(f"Encoder activation: {activation_name}")
    else:
        print("Normal activation")
        activation = None
    encoder.fc = IdLayer(activation=activation)
    #encoder.fc = IdLayer(activation=nn.Sigmoid()).to(device)
    MEAN, STD = cfg.NORMAL_PARAMS["cifar"]

    # ----------- Setup Similarity Encoder
    blackbox_dir = params["blackbox_dir"]
    encoder_ckp = params["encoder_ckp"]
    encoder_suffix = params["encoder_suffix"]
    encoder_arch_name += encoder_suffix
    if encoder_ckp is not None:
        encoder_margin = params["encoder_margin"]
        encoder_ckp = osp.join(encoder_ckp, encoder_arch_name, f"CIFAR10-margin-{encoder_margin}")
        ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
        print(f"=> Loading similarity encoder checkpoint '{ckp}'")
        checkpoint = torch.load(ckp)
        start_epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"==> Loaded encoder: arch_name: {encoder_arch_name} \n margin: {encoder_margin} \n thresh: {thresh}")

        encoder = encoder.to(device)
        encoder.eval()

        blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)
        #print(f"threshold {blackbox.thresh}, k {k}")
        blackbox.init(blackbox_dir, device, time=created_on)
    else:
        blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Set up query set
    test_transform = datasets.modelfamily_to_transforms["cifar"]['test']
    batch_size = params["batch_size"]
    num_workers = 10

    #--------- Extraction
    candidate_sets = params["testset_names"]
    conf_list = []
    for testset_name in candidate_sets:
        valid_datasets = datasets.__dict__.keys()
        if testset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[testset_name]
        dataset = datasets.__dict__[testset_name]

        #test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        trainset = dataset(train=True, transform=test_transform)
        print('=> Queryset size (training split) = {}'.format(len(trainset)))
        testset = dataset(train=False, transform=test_transform)
        print('=> Queryset size (test split) = {}'.format(len(testset)))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        total_train, correct_train = 0, 0
        for images, labels in train_loader:
            labels = labels.to(device)
            is_adv, y = blackbox(images)
            _, predicted = y.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
        print("=> Train accuracy: {:.2f}".format(correct_train/total_train))
        total_val, correct_val = 0, 0
        for images, labels in test_loader:
            labels = labels.to(device)
            is_adv, y = blackbox(images)
            _, predicted = y.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)
        print("=> Validation accuracy: {:.2f}".format(correct_val/total_val))

    params["num_detection"] = blackbox.alarm_count
    # Store arguments
    params_out_path = osp.join(log_dir, 'params_benign.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()