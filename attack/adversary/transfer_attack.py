import argparse
import os
import sys
sys.path.append('/mydata/model-extraction/prediction-poisoning/knockoffnets/')
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
sys.path.append('/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding')

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from attack import datasets
import attack.config as cfg
import modelzoo.zoo as zoo

import foolbox
from foolbox.attacks import LinfPGD as PGD
from foolbox.criteria import Misclassification, TargetedMisclassification



def main():
    parser = argparse.ArgumentParser(description='Simulate model extraction')
    # -------------------- Adversary
    parser.add_argument('--testset_name', metavar='STR', type=str, help='Testset', default="CIFAR10")
    parser.add_argument("--eps", metavar="TYPE", type=float, default=8./256)
    parser.add_argument('--targeted', action='store_true', help='Whether is targeted attack', default=False)

    # Adv model
    parser.add_argument('--model_adv_name', metavar='STR', type=str, help='Model arch of F_A', default="vgg16_bn")
    parser.add_argument("--adv_dir", metavar="PATH", type=str)

    # Blackbox
    parser.add_argument('--model_blackbox_name', metavar='STR', type=str, help='Model arch of F_A', default="vgg16_bn")
    parser.add_argument("--bb_dir", metavar="PATH", type=str, default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/vgg16_bn/checkpoint.pth.tar")
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries',
                        default=128)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)

    args = parser.parse_args()
    params = vars(args)

    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # ----------- Set up testset
    testset_name = params["testset_name"]
    valid_datasets = datasets.__dict__.keys()
    if testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform_type = 'test'
    transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    num_classes = len(testset.classes)

    loader = DataLoader(testset, batch_size=64, pin_memory=True)

    mean, std = cfg.NORMAL_PARAMS[modelfamily]
    MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1]).to(device)
    STD = torch.Tensor(std).reshape([1, 3, 1, 1]).to(device)
    def normalize(images):
        return (images-MEAN) / STD

    def denormalize(images):
        return torch.clamp(images*STD + MEAN, 0., 1.)   
    # ----------- Set up adversary model
    model_adv_name = params["model_adv_name"]

    adv_dir = params["adv_dir"]
    #imagenet pretrained
    #adv_dir = "/mydata/model-extraction/prediction-poisoning/defenses/adversary/pretrained/vgg16_bn/checkpoint.pth.tar"

    # random adv
    #adv_dir = "/mydata/model-extraction/prediction-poisoning/experiments/cifar10_seed5000_jbtop3_random_adv_adv_tran/checkpoint.pth.tar"

    #eps=0.1
    #adv_dir = "/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-10-27_23:49:56/checkpoint.pth.tar"

    #eps=0.14
    #adv_dir = "/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-10-29_03:44:29/checkpoint.pth.tar"

    #eps=0.03
    #adv_dir = "/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-10-27_20:23:54/checkpoint.pth.tar"

    #eps=0.01
    #adv_dir = "/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-10-27_10:27:42/checkpoint.pth.tar"

    # new eps=0.1
    #adv_dir="/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-11-03_21:31:48/checkpoint.pth.tar"

    #new eps=0.01
    #adv_dir="/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-11-03_21:30:13/checkpoint.pth.tar"

    # knockoffnet
    #adv_dir="/mydata/model-extraction/prediction-poisoning/experiments/knockoff_cifar10bycifar100/checkpoint.pth.tar"

    print("==> Loading adversary model...")
    adversary_model = zoo.get_net(model_adv_name, modelfamily, adv_dir,
                       num_classes=num_classes)
    adversary_model = adversary_model.to(device)
    adversary_model.eval()


    # ----------- Set up victim model
    print("==> Loading victim model...")
    model_blackbox_name = params["model_blackbox_name"]
    #bb_dir = "/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/vgg16_bn/checkpoint.pth.tar"
    bb_dir = params["bb_dir"]
    blackbox = zoo.get_net(model_blackbox_name, modelfamily, bb_dir,
                       num_classes=num_classes)
    blackbox = blackbox.to(device)
    adversary_model.eval()

    # ----------- Set up attack
    attack = PGD()
    eps = params["eps"]
    targeted = params["targeted"]
    print(f"==> PGD w/ eps={eps}, targeted: {targeted}")
    adv2adv = 0
    adv2bb = 0
    total = 0
    for images, labels in loader:
        total += images.size(0)
        images = images.to(device)
        batch_size = images.size(0)
        labels = labels.to(device)

        # Use foolbox to augment data
        if targeted:
            targets = []
            with torch.no_grad():
                pred = torch.argsort(adversary_model(images), descending=True)
                for i in range(pred.size(0)):
                    y_hat = pred[i]
                    y_hat = y_hat[y_hat != labels[i]][0]
                    targets.append(y_hat)
            targets = torch.stack(targets)
            adv_criterion = TargetedMisclassification(targets)
        else:
            adv_criterion = Misclassification(labels)
        images = denormalize(images)
        fmodel = foolbox.models.PyTorchModel(adversary_model, bounds=(0, 1), preprocessing={"mean":MEAN, "std":STD})
        #fmodel = foolbox.models.PyTorchModel(adversary_model, bounds=(images.min(), images.max()))
        _, images, is_adv = attack(fmodel, images, criterion=adv_criterion, epsilons=eps)
        images = normalize(images)

        with torch.no_grad():
            y_adv = adversary_model(images)
            y_bb = blackbox(images)
        if targeted:
            adv2adv += targets.eq(y_adv.argmax(-1)).sum().item()
            adv2bb += targets.eq(y_bb.argmax(-1)).sum().item()

        else:    
            adv2adv += labels.ne(y_adv.argmax(-1)).sum().item()
            adv2bb += labels.ne(y_bb.argmax(-1)).sum().item()

        msg = f"{adv2bb}/{total} = {int(adv2bb/total*100)}%"
        print(msg+" are adversarial to the victim")
        msg2 = f"{adv2adv}/{total} = {int(adv2adv/total*100)}%"
        print(msg2+" are adversarial to the adversary")
        print()

if __name__ == '__main__':
    main()