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
from detector_lpips import *
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
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--encoder_ckp", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--encoder_suffix", metavar="TYPE", type=str, default="")
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument("--k", metavar="TYPE", type=int, default=1)
    parser.add_argument("--target", metavar="TYPE", type=float, help="targeted FPR", default=1e-2)
    parser.add_argument("--lower", metavar="TYPE", type=float, help="binary search lowerbound", default=1e-3)
    parser.add_argument("--upper", metavar="TYPE", type=float, help="binary search upperbound", default=2.)
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
    log_suffix = params["log_suffix"]
    log_dir = params["log_dir"]
    use_lpips = params["lpips"]
    created_on = str(datetime.now()).replace(' ', '_')[:19]
    log_dir = osp.join(log_dir, created_on)
    create_dir(log_dir)

    encoder_arch_name = params["encoder_arch_name"]
    num_classes = 10
    #encoder = zoo.get_net(encoder_arch_name, "cifar", num_classes=num_classes)
    #activation_name = params['activation']
    #if activation_name == "sigmoid":
    #    activation = nn.Sigmoid()
    #    print(f"Encoder activation: {activation_name}")
    #else:
    #    print("Normal activation")
    #    activation = None
    #encoder.fc = IdLayer(activation=activation)
    MEAN, STD = cfg.NORMAL_PARAMS["cifar"]

    # ----------- Setup Similarity Encoder
    blackbox_dir = params["blackbox_dir"]
    encoder_ckp = params["encoder_ckp"]
    encoder_suffix = params["encoder_suffix"]
    candidate_sets = params["testset_names"]
    # setup similarity encoder
    if use_lpips:
        blackbox = LpipsDetector(k, thresh, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on)
    elif encoder_ckp:
        encoder_arch_name = params["encoder_arch_name"]
        #encoder = zoo.get_net(encoder_arch_name, modelfamily, num_classes=num_classes)
        testset_name = candidate_sets[0]
        modelfamily = datasets.dataset_to_modelfamily[testset_name]
        encoder = zoo.get_net(encoder_arch_name, modelfamily, num_classes=num_classes)
        activation_name = params['activation']
        if activation_name == "sigmoid":
            activation = nn.Sigmoid()
            print(f"Encoder activation: {activation_name}")
        else:
            print("Normal activation")
            activation = None

        encoder.fc = IdLayer(activation=activation).to(device)

        encoder_ckp = params["encoder_ckp"]
        encoder_suffix = params["encoder_suffix"]
        encoder_arch_name += encoder_suffix

        encoder_margin = params["encoder_margin"]
        encoder_ckp = osp.join(encoder_ckp, encoder_arch_name, f"{testset_name}-margin-{encoder_margin}")
        ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
        print(f"=> Loading similarity encoder checkpoint '{ckp}'")
        checkpoint = torch.load(ckp)
        start_epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        #encoder.fc = IdLayer(activation=activation)
        encoder = encoder.to(device)
        encoder.eval()
        #print(f"==> Loaded encoder: arch_name: {encoder_arch_name} \n margin: {encoder_margin} \n thresh: {thresh}")

        #blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)
        #blackbox.init(blackbox_dir, device, time=created_on)
    else:
        blackbox = Blackbox.from_modeldir(blackbox_dir, device)
    # ----------- Set up query set
    batch_size = params["batch_size"]

    #--------- Binary Search for threshold
    lower, upper = params["lower"], params["upper"]
    fpr = 1.
    target = params["target"]

    while abs(fpr - target) > 1e-4: # FPR = 0.1 +- 0.01% 
        thresh = (lower+upper)/2.
        print(f"Current FPR: {fpr}%")
        print(f"Searching from [{lower}, {upper}]")
        print(f"Calculating FPR for thresh {thresh}...")
        blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on)
        total_input = 0
    
        for testset_name in candidate_sets:
            valid_datasets = datasets.__dict__.keys()
            if testset_name not in valid_datasets:
                raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
            modelfamily = datasets.dataset_to_modelfamily[testset_name]
            dataset = datasets.__dict__[testset_name]

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

            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

            total_train, correct_train = 0, 0
            for images, labels in train_loader:
                labels = labels.to(device)
                y = blackbox(images)
                _, predicted = y.max(1)
                correct_train += predicted.eq(labels).sum().item()
                total_train += labels.size(0)
                total_input += labels.size(0)
            print("=> Train accuracy: {:.2f}".format(correct_train/total_train))
            total_val, correct_val = 0, 0
            for images, labels in test_loader:
                labels = labels.to(device)
                y = blackbox(images)
                _, predicted = y.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)
                total_input += labels.size(0)
            print("=> Validation accuracy: {:.2f}".format(correct_val/total_val))

        fpr = 100*blackbox.alarm_count/(total_input)

        if fpr > target:
            upper = thresh
        else:
            lower = thresh
        print()

    params["thresh"] = thresh
    params["num_detection"] = blackbox.alarm_count
    params["total input"] = total_input
    params["false_positive_rate"] = f"{100*blackbox.alarm_count/(total_input)}%"
    print(f"false positive rate: {100*blackbox.alarm_count/total_input}%")
    print(f"threshold: {thresh}")
    # Store arguments
    params_out_path = osp.join(log_dir, 'params_benign.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)
    query_dist_out_path = osp.join(log_dir, 'query_dist.pt')
    torch.save(blackbox.query_dist, query_dist_out_path)

if __name__ == '__main__':
    main()