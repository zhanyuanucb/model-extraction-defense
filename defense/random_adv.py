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
    parser = argparse.ArgumentParser(description='Simulate model extraction')
    parser.add_argument("--model_name", metavar="TYPE", type=str, default="resnet18")
    parser.add_argument("--device_id", metavar="TYPE", type=int, default=1)
    parser.add_argument("--jid", metavar="TYPE", type=int, default=1)
    parser.add_argument("--num_classes", metavar="TYPE", type=int, default=10)
    parser.add_argument("--out_root", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/")
    parser.add_argument("--batch_size", metavar="TYPE", type=int, default=32)
    parser.add_argument("--eps", metavar="TYPE", type=float, help="JDA step size", default=0.01)
    parser.add_argument("--steps", metavar="TYPE", type=int, help="number of JDA steps", default=7)
    parser.add_argument("--phi", metavar="TYPE", type=int, help="number of extraction iterations", default=6)
    parser.add_argument("--alt_t", metavar="TYPE", type=int, help="alternate period of step size sign", default=None)
    parser.add_argument("--epochs", metavar="TYPE", type=int, help="extraction training epochs", default=10)
    parser.add_argument("--momentum", metavar="TYPE", type=float, help="multi-step JDA momentum", default=0.7)
    parser.add_argument("--blackbox_dir", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28")
    parser.add_argument("--blinders_dir", metavar="PATH", type=str,
                        default=None)
    parser.add_argument("--r", metavar="PATH", type=float, help="params of random transform blinders",
                        default=None)
    parser.add_argument("--trainset_name", metavar="TYPE", type=str, default="CINIC10")
    parser.add_argument("--testset_name", metavar="TYPE", type=str, default="CIFAR10")
    parser.add_argument("--optimizer_name", metavar="TYPE", type=str, default="adam")
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument("--encoder_ckp", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/")
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--k", metavar="TYPE", type=int, default=10)
    parser.add_argument("--thresh", metavar="TYPE", type=float, help="detector threshold", default=0.0397684188708663)
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="testing")
    parser.add_argument("--params_search", action="store_true")
    parser.add_argument("--random_adv", action="store_true")
    args = parser.parse_args()
    params = vars(args)

    # ------------ Start
    #device = torch.device("cuda:1" if use_cuda else "cpu")
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19]
    created_on = params['created_on']
    out_root = params["out_root"]

    ckp_out_root = osp.join(out_root, created_on)
    if not osp.exists(ckp_out_root):
        os.mkdir(ckp_out_root)
    # ----------- Initialize Detector
    k = params["k"]
    thresh = params["thresh"]
    log_suffix = params["log_suffix"]
    log_dir = ckp_out_root
    trainset_name = params["trainset_name"]
    testset_name = params["testset_name"]
    encoder_arch_name = params["encoder_arch_name"]
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    num_classes = 10
    encoder = zoo.get_net(encoder_arch_name, modelfamily, num_classes=num_classes)
    MEAN, STD = cfg.NORMAL_PARAMS[modelfamily]

    # setup similarity encoder
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

    # ----------- Initialize adversary model
    model_name = params["model_name"]
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    model = model.to(device)

    # ----------- Initialize Adversary
    nworkers = 10
    # attack parameters
    eps = params["eps"]
    steps= params["steps"]
    momentum= params["momentum"]

    # set up query blinding
    blinders_dir = params["blinders_dir"]
    if blinders_dir is not None:
        blinders_ckp = osp.join(blinders_dir, "checkpoint.blind.pth.tar")
        if osp.isfile(blinders_ckp):
            blinders_noise_fn = blinders_transforms.get_gaussian_noise(device=device, r=0.095)
            auto_encoder = AutoencoderBlinders(blinders_noise_fn)
            print("=> Loading auto-encoder checkpoint '{}'".format(blinders_ckp))
            checkpoint = torch.load(blinders_ckp, map_location=device)
            start_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['best_loss']
            auto_encoder.load_state_dict(checkpoint['state_dict'])
            print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            print(f"===> Best val loss: {best_test_loss}")
            auto_encoder = auto_encoder.to(device)
            auto_encoder.eval()
        else:
            r = params['r']
            print("===> no checkpoint found at '{}'".format(blinders_ckp))
            print("===> Loading random transform query blinding...")
            auto_encoder = eval(f"blinders_transforms.{blinders_dir}")(device=device, r=r)
    else:
        auto_encoder = None

    adversary = MultiStepJDA(model, blackbox, MEAN, STD, device, blinders_fn=auto_encoder, eps=eps, steps=steps, momentum=momentum) 

#    # ----------- Set up seedset
#    seedset_path = osp.join(params["seedset_dir"], 'seed.pt')
#    images_sub, labels_sub = torch.load(seedset_path)
#    seedset_samples = [images_sub, labels_sub]
#    num_classes = seedset_samples[1][0].size(0)
#    print('=> found transfer set with {} samples, {} classes'.format(seedset_samples[0].size(0), num_classes))

    # ----------- Set up trainset
    valid_datasets = datasets.__dict__.keys()
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train'] # test2 has no normalization
    if trainset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[trainset_name]
    trainset = dataset(train=True, transform=train_transform)

    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up testset
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test'] # test2 has no normalization
    if testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[testset_name]
    testset = dataset(train=False, transform=test_transform)

    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up seed images
    print('=> Training at budget = {}'.format(len(trainset)))

    optimizer_name = params["optimizer_name"]
    optimizer = model_utils.get_optimizer(model.parameters(), optimizer_name)

    criterion_train = model_utils.soft_cross_entropy

    #--------- Extraction
    random_adv = params["random_adv"]
    phi = params["phi"]
    alt_t = params["alt_t"]
    steps = params["steps"]
    batch_size = params["batch_size"]
    budget = (phi+1)*len(substitute_set)
    checkpoint_suffix = '.budget{}'.format(budget)
    testloader = testset
    epochs = params["epochs"]
    num_workers = 10
    #train_loader = DataLoader(substitute_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    aug_loader = DataLoader(substitute_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    substitute_out_path = osp.join(out_root, f"substitute_set.pt")
    for p in range(1, phi+1):
        if alt_t: # Apply periodic step size
            adversary.JDA.lam *= (-1)**(p//alt_t)

        if random_adv:
            assert phi == 1, "Random Adversary only needs 1 extraction epoch"
            substitute_set = ImageTensorSet(seedset_samples) # baseline
        else:
            images_aug, labels_aug = adversary(aug_loader)
            nxt_aug_samples = [images_aug.clone(), labels_aug.clone()]
            nxt_aug_set = ImageTensorSet(nxt_aug_samples)
            aug_loader = DataLoader(nxt_aug_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            images_sub = torch.cat([images_sub, images_aug.clone()])
            labels_sub = torch.cat([labels_sub, labels_aug.clone()])
            substitute_samples = [images_sub, labels_sub]
            substitute_set = ImageTensorSet(substitute_samples)

            torch.save(substitute_samples, substitute_out_path)
            print('=> substitute set ({} samples) written to: {}'.format(substitute_samples[0].size(0), substitute_out_path))

        print(f"Substitute training epoch {p}")
        print(f"Current size of the substitute set {len(substitute_set)}")
        best_test_acc, train_loader = model_utils.train_model(model, substitute_set, ckp_out_root, batch_size=batch_size, epochs=epochs, testset=testloader, criterion_train=criterion_train,
                                                  checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)

    # Store arguments
    params['budget'] = images_sub.size(0)
    params_out_path = osp.join(ckp_out_root, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    if params["params_search"]:
        jid = params["jid"]
        search_log_path = f"./params_search_{jid}.log.tsv"
        if not osp.exists(search_log_path):
            with open(search_log_path, 'w') as log:
                columns = ["created_on", "best_test_acc"]
                log.write('\t'.join(columns) + '\n')
            print(f"Created log file at {search_log_path}")

        with open(search_log_path, 'a') as log:
            log.write('\t'.join([created_on, str(best_test_acc)]) + '\n')

if __name__ == '__main__':
    main()