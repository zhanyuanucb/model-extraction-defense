import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os
import os.path as osp
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import transforms as tvtransforms
from attack import datasets
import modelzoo.zoo as zoo
import attack.utils.model as model_utils
import attack.config as cfg
import blind as blind_utils
import transforms as mytransforms
from datetime import datetime

class BlindLoss(nn.Module):

    def __init__(self, blind, blackbox, random_transform, d=10, c=1, mean=None, std=None):
        super(BlindLoss, self).__init__()
        self.blind = blind
        self.blackbox = blackbox
        self.random_transform = random_transform
        self.d = d
        self.c = c
        self.MEAN = mean
        self.STD = std
        self.normalize = self.MEAN is not None and self.STD is not None

    def forward(self, x):

        if self.normalize:
            x_norm = (x - self.MEAN) / self.STD
            y = self.blackbox(x_norm)
        else:
            y = self.blackbox(x)
        
        x0 = self.random_transform(x)
        _, x_hat = self.blind(x0)

        if self.normalize:
            x_hat_norm = (x_hat - self.MEAN) / self.STD
            y_hat = self.blackbox(x_hat_norm)
        else:
            y_hat = self.blackbox(x_hat)

        H = model_utils.soft_cross_entropy(y, y_hat)
        x1 = self.random_transform(x)
        x2 = self.random_transform(x)
        _, x1_hat = self.blind(x1)
        _, x2_hat = self.blind(x2)
        C = torch.clamp(torch.norm(x1_hat-x2_hat)**2, 0., self.d**2)
        return H + self.c*C


def main():
    parser = argparse.ArgumentParser(description='Train similarity encoder')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/auto_encoder")
    parser.add_argument('--ckp_dir', metavar='PATH', type=str,
                        help='Destination directory to store trained model', default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28")
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of dataset', default='CIFAR10')
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="wrn28")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('--train_epochs', metavar='TYPE', type=int, help='Training epochs', default=10)
    parser.add_argument('--optimizer_name', metavar='TYPE', type=str, help='Optimizer name', default="adam")
    parser.add_argument('--ckpt_suffix', metavar='TYPE', type=str, default="")
    parser.add_argument('--load_pretrained', action='store_true')

    # ----------- Other params
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker processes to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
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

    # ------------ Set up blackbox
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
    auto_encoder = blind_utils.Autoencoder()
    auto_encoder = auto_encoder.to(device)
    # ------------

    # ------------ Set up random transformation
    T = [mytransforms.get_random_gaussian_pt(device=device, max_sigma=0.1),
         mytransforms.get_random_rotate_kornia(max_deg=22.5),
         mytransforms.get_random_contrast_pt(device=device, min_alpha=0.9, max_alpha=1.4),
         mytransforms.get_random_brightness_pt(device=device, min_beta=-0.05, max_beta=0.05)]

    random_transform = mytransforms.RandomTransform(T)
    # ------------

    # ------------ Set up loss function
    blindloss = BlindLoss(auto_encoder, model, random_transform, mean=MEAN, std=STD)
    # ------------

    train_epochs = params['train_epochs']
    optimizer_name = params["optimizer_name"]
    optimizer = model_utils.get_optimizer(auto_encoder.parameters(), optimizer_name)
    checkpoint_suffix = params["ckpt_suffix"]
    ckp_dir = params["ckp_dir"]

    checkpoint_suffix = ".blind"
    if not osp.exists(out_path):
        os.mkdir(out_path)
    blind_utils.train_model(model, trainset, out_path, epochs=train_epochs, testset=valset,
                            criterion_train=blindloss, criterion_test=blindloss,
                            checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, f'params_train{checkpoint_suffix}.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()