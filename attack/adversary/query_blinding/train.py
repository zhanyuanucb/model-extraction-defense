import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import json
import os
import os.path as osp
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as tvtransforms
from attack import datasets
import modelzoo.zoo as zoo
import attack.utils.model as model_utils
import attack.utils.utils as os_utils
import attack.config as cfg
import blinders
import transforms as mytransforms
from datetime import datetime

class BlindersLoss(nn.Module):

    def __init__(self, auto_encoder, f, d=10, c=1, mean=None, std=None):
        super(BlindersLoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        self.auto_encoder = auto_encoder
        self.f = f
        self.d = d
        self.c = c
        self.MEAN = mean
        self.STD = std
        self.normalize = self.MEAN is not None and self.STD is not None

    def forward(self, x):

        # -------- Target label
        if self.normalize:
            x_norm = (x - self.MEAN) / self.STD
            y = self.f(x_norm)
        else:
            y = self.f(x)
        target = F.softmax(y, dim=1)
        # --------
        
        # -------- Query blinding
        x0_hat = self.auto_encoder(x)
        # --------

        # -------- Output label
        if self.normalize:
            x0_hat_norm = (x0_hat - self.MEAN) / self.STD
            y_hat = self.f(x0_hat_norm)
        else:
            y_hat = self.f(x0_hat)
        # --------

        H = model_utils.soft_cross_entropy(y_hat, target)

        x1_hat = self.auto_encoder(x)
        x2_hat = self.auto_encoder(x)
        x_diff = (x1_hat - x2_hat).view(x.size(0), -1)
        C = torch.clamp(torch.mean(torch.norm(x_diff, p=2, dim=1))**2, 0, self.d**2)

        return H - self.c*C

class AutoEncoderBCELoss(nn.Module):

    def __init__(self, AutoEncoder):
        super(AutoEncoderBCELoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        self.AutoEncoder = AutoEncoder

    def forward(self, x):
        _, x_t = self.AutoEncoder(x)
        return self.BCELoss(x_t, x)

def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind")
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/f")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of dataset', default='CIFAR10')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="wrn28")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=64)
    parser.add_argument('--lr', metavar='TYPE', type=float, default=0.01)
    parser.add_argument('--train_epochs', metavar='TYPE', type=int, help='Training epochs', default=10)
    parser.add_argument('--optimizer_name', metavar='TYPE', type=str, help='Optimizer name', default="adam")
    parser.add_argument('--ckpt_suffix', metavar='TYPE', type=str, default="")
    parser.add_argument('--resume', metavar="PATH", type=str, default=None)
    parser.add_argument('--load_phase1', action='store_true')

    # ----------- Other params
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker processes to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_root = params['out_dir']
    #torch.manual_seed(cfg.DEFAULT_SEED)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    # ----------- Set up dataset
    dataset_name = params['dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['test2']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test2']
    #random_transform = transform_utils.RandomTransforms(modelfamily=modelfamily)
    trainset = datasets.__dict__[dataset_name](train=True, transform=tvtransforms.ToTensor()) # Augment data while training
    valset = datasets.__dict__[dataset_name](train=False, transform=tvtransforms.ToTensor())

    # ------------ Set up based classifier
    model_dir = params['ckp_dir']
    model_path = osp.join(model_dir, "checkpoint.pth.tar")
    model_name = params['model_name']
    num_classes = params['num_classes']
    dataset_name = params['dataset_name']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    MEAN, STD = cfg.NORMAL_PARAMS[modelfamily]
    MEAN, STD = torch.Tensor(MEAN).reshape([1, 3, 1, 1]), torch.Tensor(STD).reshape([1, 3, 1, 1])
    MEAN, STD = MEAN.to(device), STD.to(device)
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)
    if osp.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Best val acc: {best_test_acc}%")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        exit(1)
    model = model.to(device)
    # ------------

    # ------------ Set up Auto-encoder
    blinders_fn = mytransforms.get_gaussian_noise(device=device, sigma=0.095)
    auto_encoder = blinders.AutoencoderBlinders(blinders_fn)
    auto_encoder = auto_encoder.to(device)
    # ------------

    # ------------ Set up loss function
    blindloss = BlindersLoss(auto_encoder, model, mean=MEAN, std=STD)
    # ------------

    # ------------ Set up training
    train_epochs = params['train_epochs']
    optimizer_name = params["optimizer_name"]
    checkpoint_suffix = params["ckpt_suffix"]
    ckp_dir = params["ckp_dir"]
    resume = None if params["resume"] == "None" else params["resume"]

    checkpoint_suffix = ".blind"


    os_utils.create_dir(out_root)

    # --------------------- Phase 1
    if params["load_phase1"]:
        optimizer = model_utils.get_optimizer(auto_encoder.parameters(), optimizer_name)

        out_path = osp.join(out_root, "phase1")
        phase1_ckp_path = osp.join(out_path, f"checkpoint{checkpoint_suffix}.pth.tar")
        if osp.isfile(phase1_ckp_path):
            print("=> Phase 1: loading checkpoint '{}'".format(phase1_ckp_path))
            checkpoint = torch.load(phase1_ckp_path)
            start_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['best_loss']
            auto_encoder.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            print(f"=> Best val loss: {best_test_loss}%")
        else:
            print("=> no checkpoint found at '{}'".format(phase1_ckp_path))
            print("Start phase 1 training...")

            criterion = AutoEncoderBCELoss(auto_encoder)
            os_utils.create_dir(out_path)
            blind_utils.train_model(auto_encoder, trainset, out_path, epochs=train_epochs, testset=valset,
                                    criterion_train=criterion, criterion_test=criterion, resume=resume,
                                    checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)

            params['created_on'] = str(datetime.now())
            params_out_path = osp.join(out_path, f'params_train{checkpoint_suffix}.json')
            with open(params_out_path, 'w') as jf:
                json.dump(params, jf, indent=True)

    # -------------------- Phase 2
    print("Start phase 2 training...")
    batch_size = params["batch_size"]
    num_workers = params["nworkers"]
    lr = params["lr"]
    optimizer = model_utils.get_optimizer(auto_encoder.parameters(), optimizer_name, lr=lr)
    criterion = blindloss
    out_path = osp.join(out_root, "phase2")
    os_utils.create_dir(out_path)
    blinders.train_model(auto_encoder, trainset, out_path, batch_size=batch_size, epochs=train_epochs, testset=valset,
                            criterion_train=criterion, criterion_test=criterion, resume=resume,
                            checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, f'params_train{checkpoint_suffix}.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)
   
if __name__ == '__main__':
    main()