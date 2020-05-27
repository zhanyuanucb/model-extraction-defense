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
from utils import ImageTensorSet

__author__ = "Zhanyuan Zhang"
__author_email__ = "zhang_zhanyuan@berkeley.edu"
__reference__ = "https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/train.py"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Simulate benign users')
    parser.add_argument("--num_classes", metavar="TYPE", type=int, default=10)
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=32)
    parser.add_argument("--blackbox_dir", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28")
    parser.add_argument("-l", "--testset_names", nargs='+', type=str, required=True)
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--encoder_ckp", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--k", metavar="TYPE", type=int, default=5)
    parser.add_argument("--thresh", metavar="TYPE", type=float, help="detector threshold", default=0.049665371380746365)
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="benign")
    parser.add_argument("--log_dir", metavar="PATH", type=str,
                        default="./")
    parser.add_argument("--return_conf_max", action="store_true")
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
    return_conf_max = params["return_conf_max"]
    num_classes = 10
    encoder = zoo.get_net(encoder_arch_name, "cifar", num_classes=num_classes)
    MEAN, STD = cfg.NORMAL_PARAMS["cifar"]

    # ----------- Setup Similarity Encoder
    blackbox_dir = params["blackbox_dir"]
    encoder_ckp = params["encoder_ckp"]
    if encoder_ckp is not None:
        encoder_margin = params["encoder_margin"]
        encoder_ckp = osp.join(encoder_ckp, encoder_arch_name, f"CIFAR10-margin-{encoder_margin}")
        ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
        print(f"=> Loading similarity encoder checkpoint '{ckp}'")
        checkpoint = torch.load(ckp)
        start_epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        encoder = encoder.to(device)

        blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir, return_max_conf=return_conf_max)
        blackbox.init(blackbox_dir, device, time=created_on)
    else:
        blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Set up query set
    test_transform = datasets.modelfamily_to_transforms["cifar"]['test']
    batch_size = params["batch_size"]
    num_workers = 10

#    testset_name = "CIFAR10"
#    query_train_dir = cfg.dataset2dir[testset_name]["train"]
#    query_train_images, query_train_labels = torch.load(query_train_dir)
#    #query_train_images = query_train_images.permute(0, 3, 1, 2)
#    query_train_samples = (query_train_images, query_train_labels)
#    query_trainset = ImageTensorSet(query_train_samples, transform=test_transform)
#    queryloader_train = DataLoader(query_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#    print('=> Queryset size (training split) = {}'.format(len(query_trainset)))
#    query_val_dir = cfg.dataset2dir[testset_name]["test"]
#    query_val_images, query_val_labels = torch.load(query_val_dir)
#    #query_val_images = query_val_images.permute(0, 3, 1, 2)
#    query_val_samples = (query_val_images, query_val_labels)
#    query_valset = ImageTensorSet(query_val_samples, transform=test_transform)
#    queryloader_val = DataLoader(query_valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#    print('=> Queryset size (validation split) = {}'.format(len(query_valset)))
#    if return_conf_max:
#        conf_list = []
#        total_train, correct_train = 0, 0
#        for images, labels in queryloader_train:
#            labels = labels.to(device)
#            is_adv, y, conf_max = blackbox(images)
#            _, predicted = y.max(1)
#            correct_train += predicted.eq(labels).sum().item()
#            total_train += labels.size(0)
#            conf_list.append(conf_max.clone())
#        print("=> Train accuracy: {:.2f}".format(correct_train/total_train))
#        total_val, correct_val = 0, 0
#        for images, labels in queryloader_val:
#            labels = labels.to(device)
#            is_adv, y, conf_max = blackbox(images)
#            _, predicted = y.max(1)
#            correct_val += predicted.eq(labels).sum().item()
#            total_val += labels.size(0)
#            conf_list.append(conf_max.clone())
#        print("=> Validation accuracy: {:.2f}".format(correct_val/total_val))
#    conf_list = torch.cat(conf_list).cpu().numpy()
#    plt.hist(conf_list, bins=50, density=True)
#    plt.title(f"Histogram of blackbox prediction confidence (benign user)")
#    plt.savefig(osp.join(log_dir, 'conf_hist.png'))
#    torch.save(conf_list, osp.join(log_dir, 'conf_list.pkl'))


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

        if return_conf_max:
            total_train, correct_train = 0, 0
            for images, labels in train_loader:
                is_adv, y, conf_max = blackbox(images)
                labels = labels.to(device)
                is_adv, y, conf_max = blackbox(images)
                _, predicted = y.max(1)
                correct_train += predicted.eq(labels).sum().item()
                total_train += labels.size(0)
                conf_list.append(conf_max.clone())
            print("=> Train accuracy: {:.2f}".format(correct_train/total_train))
            total_val, correct_val = 0, 0
            for images, labels in test_loader:
                is_adv, y, conf_max = blackbox(images)
                labels = labels.to(device)
                is_adv, y, conf_max = blackbox(images)
                _, predicted = y.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)
                conf_list.append(conf_max.clone())
            print("=> Validation accuracy: {:.2f}".format(correct_val/total_val))
        else:
            for images, labels in train_loader:
                is_adv, y = blackbox(images)
            for images, labels in test_loader:
                is_adv, y = blackbox(images)
    conf_list = torch.cat(conf_list).cpu().numpy()
    plt.hist(conf_list, bins=50, density=True)
    plt.title(f"Histogram of blackbox prediction confidence (benign user)")
    plt.savefig(osp.join(log_dir, 'conf_hist.png'))
    torch.save(conf_list, osp.join(log_dir, 'conf_list.pkl'))

    # Store arguments
    params_out_path = osp.join(log_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()