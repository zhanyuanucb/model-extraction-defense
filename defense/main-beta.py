import argparse
import json
import os
import sys
sys.path.append('../')
sys.path.append('../attack/adversary/query_blinding')
sys.path.append('./likelihood_estimation/pytorch-generative')
import os.path as osp
from datetime import datetime
import torch

from defense.likelihood_estimation.vq_vae import VQVAE
import attack.config as cfg
import attack.utils.model as model_utils
from attack.adversary.detector_adv import AdvDetector
from attack import datasets
import modelzoo.zoo as zoo
from detector import *
#from detector_lpips import *
#from attack.adversary.jda import MultiStepJDA, AdvDA
from attack.adversary.jacobian import *
from attack.adversary.query_blinding.blinders import AutoencoderBlinders
import attack.adversary.query_blinding.transforms as blinders_transforms
from utils import ImageTensorSet, IdLayer

import pytorch_generative as pg
from likelihood_estimation.pytorch_vqvae.modules import VectorQuantizedVAE, GatedPixelCNN

__author = "Zhanyuan Zhang"
__author_email__ = "zhang_zhanyuan@berkeley.edu"
__reference__ = "https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/train.py"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Simulate model extraction')
    # -------------------- Adversary
    parser.add_argument("--blinders_dir", metavar="PATH", type=str,
                        default=None)
    parser.add_argument("--r", metavar="TYPE", type=str, help="params of random transform blinders",
                        default="low")
    parser.add_argument('--policy', metavar='PI', type=str, help='Policy to use while training')
    parser.add_argument('--model_adv', metavar='STR', type=str, help='Model arch of F_A', default=None)
    parser.add_argument('--pretrained', metavar='STR', type=str, help='Assumption of F_A', default=None)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries',
                        default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument('--aug_batch_size', metavar='TYPE', type=int, help='Batch size of queries',
                        default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument("--ema_decay", metavar="TYPE", type=float, default=-1.)
    parser.add_argument('--budget', metavar='N', type=int, help='Query limit to blackbox', default=35000)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Data for seed images', required=True, default="CIFAR10")
    parser.add_argument('--seedsize', metavar='N', type=int, help='Size of seed set', default=500)
    parser.add_argument('--train_transform', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument('--random_adv', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument('--adv_transform', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument('--testset', metavar='TYPE', type=str, help='Blackbox testset (P_V(X))', required=True)
    parser.add_argument("--num_classes", metavar="TYPE", type=int, default=10)
    parser.add_argument("--fb_eps_factor", metavar="TYPE", type=float, default=1.)
    parser.add_argument("--eps", metavar="TYPE", type=float, default=0.1)
    parser.add_argument('--num_steps', metavar='N', type=int, help='# steps', default=8)
    parser.add_argument('--rho', metavar='N', type=int, help='# Data Augmentation Steps', default=20)
    parser.add_argument('--sigma', metavar='N', type=int, help='Reservoir sampling beyond these many epochs', default=-1)
    parser.add_argument('--kappa', metavar='N', type=int, help='Size of reservoir', default=None)
    parser.add_argument('--take_lastk', metavar='N', type=int, help='Size of reservoir', default=-1)
    parser.add_argument('--tau', metavar='N', type=int,
                        help='Iteration period after which step size is multiplied by -1', default=None)
    parser.add_argument('--train_epochs', metavar='N', type=int, help='# Epochs to train model', default=20)

    # -------------------- Blackbox/Detector
    parser.add_argument("--blackbox_dir", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28_2")
    parser.add_argument('--output_type', metavar='TYPE', type=str, help='Output type of Blackbox ["one_hot", "prob"]', default="one_hot")
    parser.add_argument("--T", metavar="TYPE", type=float, default=1.)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument('--activation', metavar='TYPE', type=str, help='Activation name', default=None)
    parser.add_argument("--encoder_suffix", metavar="TYPE", type=str, default="")
    parser.add_argument("--encoder_ckpt", metavar="PATH", type=str,
                        default=None)
    parser.add_argument("--vqvae_ckpt", metavar="PATH", type=str, default=None)
    parser.add_argument("--pixelcnn_ckpt", metavar="PATH", type=str, default=None)
    parser.add_argument("--resume", metavar="PATH", type=str,
                        default=None)
    parser.add_argument("--encoder_margin", metavar="TYPE", type=float, default=3.2)
    parser.add_argument("--k", metavar="TYPE", type=int, default=1)
    parser.add_argument("--num_clusters", metavar="TYPE", type=int, default=50)
    parser.add_argument("--thresh", metavar="TYPE", type=float, help="detector threshold", default=0.12986328125)
    parser.add_argument("--envelope_size", metavar="TYPE", type=float, help="detector envelope size", default=0.3763033370971679)
    #parser.add_argument('--adaptive_adv', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument("--adaptive_adv_ckpt", metavar="PATH", type=str, default=None)
    parser.add_argument("--adaptive_thresh", metavar="PATH", type=float, default=None)
    parser.add_argument("--adv_encoder_arch_name", metavar="TYPE", type=str, default="simnet")
    parser.add_argument('--binary_search', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument('--featurefool', action='store_true', help='Perform data augmentation', default=False)
    parser.add_argument("--foolbox_alg", metavar="TYPE", type=str, help="['pgd', 'cw_l2']", default="pgd")

    # -------------------- Other params
    parser.add_argument("--log_suffix", metavar="TYPE", type=str, default="testing")
    parser.add_argument("--out_root", metavar="PATH", type=str,
                        default="/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/")
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)

    args = parser.parse_args()
    params = vars(args)

    # ------------ Start
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    params['created_on'] = str(datetime.now()).replace(' ', '_')[:19].replace(":", "-")
    log_suffix = params["log_suffix"]
    created_on = params['created_on']
    out_root = params["out_root"]

    ckp_out_root = osp.join(out_root, created_on+"-"+log_suffix)
    if not osp.exists(ckp_out_root):
        os.mkdir(ckp_out_root)

    # ----------- Set up testset
    testset_name = params['testset']
    valid_datasets = datasets.__dict__.keys()
    if testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform_type = 'test'
    transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    #num_classes = len(testset.classes)
    num_classes = params['num_classes']

    # ----------- Set up queryset
    # Load seedset from ImageFolder
    if params['seedsize'] == 500 and params['queryset']=="CIFAR10":
        folder_dir = "/mydata/model-extraction/data/cifar10_balanced_subset500"
        #folder_dir = "/mydata/model-extraction/data/cifar10_balanced_subset5000"
        seedset = ImageFolder(folder_dir, transform=transform)
        print(f"=> Load CIFAR10 seedset of size 500 from {folder_dir}")
    else:
        queryset_name = params['queryset']
        print(f"=> Prepare seedset from {queryset_name}")
        valid_datasets = datasets.__dict__.keys()
        if queryset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[queryset_name]
        transform_type = 'train' if params['train_transform'] else 'test'
        if params['train_transform']:
            print('=> Using data augmentation while querying')
        transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
        try:
            queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
        except TypeError as e:
            queryset = datasets.__dict__[queryset_name](split="train", transform=transform)

        # Use a subset of queryset
        subset_idxs = np.random.choice(range(len(queryset)), size=params['seedsize'], replace=False)
        seedset = Subset(queryset, subset_idxs)

    # ----------- Initialize Detector
    k = params["k"]
    num_clusters = params["num_clusters"]
    thresh = params["thresh"]
    log_dir = ckp_out_root
    MEAN, STD = cfg.NORMAL_PARAMS[modelfamily]
    modelfamily = "cifar" # Performing Jbtop3 on several datasets

    use_lpips = params["lpips"]
    blackbox_dir = params["blackbox_dir"]
    output_type = params["output_type"]
    T = params["T"]
    encoder_ckpt = params["encoder_ckpt"]
    encoder_margin = params["encoder_margin"]
    vqvae_ckpt = params["vqvae_ckpt"]
    pixelcnn_ckpt = params["pixelcnn_ckpt"]

    # setup similarity encoder
    if use_lpips:
        blackbox = LpipsDetector(k, thresh, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on, output_type=output_type, T=T)
    elif encoder_ckpt:
        encoder_arch_name = params["encoder_arch_name"]
        encoder = zoo.get_net(encoder_arch_name, modelfamily, num_classes=num_classes)
        encoder_suffix = params["encoder_suffix"]
        encoder_arch_name += encoder_suffix
        encoder_ckpt = osp.join(encoder_ckpt, encoder_arch_name, f"{testset_name}-margin-{encoder_margin}")
        activation_name = params['activation']
        if activation_name == "sigmoid":
            activation = nn.Sigmoid()
            print(f"Encoder activation: {activation_name}")
        else:
            print("Normal activation")
            activation = None

        encoder.fc = IdLayer(activation=activation).to(device)

        ckp = osp.join(encoder_ckpt, f"checkpoint.sim-{encoder_margin}.pth.tar")
        print(f"=> Loading similarity encoder checkpoint '{ckp}'")
        checkpoint = torch.load(ckp)
        encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        #encoder.fc = IdLayer(activation=activation)
        encoder = encoder.to(device)
        encoder.eval()
        print(f"==> Loaded encoder: \n arch_name: {encoder_arch_name} \n margin: {encoder_margin} \n thresh: {thresh}")

        #blackbox = Detector(k, thresh, encoder, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)
        #blackbox = Detector(k, thresh, encoder, MEAN, STD, num_clusters=num_classes, log_suffix=log_suffix, log_dir=log_dir)
        blackbox = Detector(k, thresh, encoder, MEAN, STD, num_clusters=num_clusters, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on, output_type=output_type, T=T)

    elif vqvae_ckpt:
        ###################
        # From pytorch-generative
        ###################
        num_hiddens = 128
        num_residual_hiddens = 32
        num_residual_layers = 2
        embedding_dim = 64
        num_embeddings = 512
        commitment_cost = 0.25
        decay = 0.99
        encoder = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(device)
        encoder.load_ckpt(vqvae_ckpt)
        encoder.eval()

        #####################
        # From pytorch_vqvae
        #####################
        #num_channels = 3
        #hidden_size_vae = 256
        #num_embeddings = 512
        #encoder = VectorQuantizedVAE(num_channels, hidden_size_vae, num_embeddings).to(device)
        #encoder.load_state_dict(torch.load(vqvae_ckpt))
        #encoder.eval()

        if pixelcnn_ckpt:
            ###################
            # From pytorch-generative
            ###################
            #get_gated_pixelcnn = pg.models.gated_pixel_cnn
            #in_channels= 1
            #out_channels = num_embeddings
            #n_gated = 11 #num_layers_pixelcnn = 12
            #gated_channels = 32 #fmaps_pixelcnn = 32
            #head_channels = 32
            #prior = get_gated_pixelcnn.GatedPixelCNN(in_channels=in_channels, 
            #                                out_channels=out_channels,
            #                                n_gated=n_gated,
            #                                gated_channels=gated_channels, 
            #                                head_channels=head_channels).to(device)
            #prior.load_ckpt(pixelcnn_ckpt)
            #prior.eval()
            #blackbox = ELBODetector(k, thresh, encoder, prior, MEAN, STD, num_clusters=num_classes, log_suffix=log_suffix, log_dir=log_dir)
            
            #####################
            # From pytorch_vqvae
            #####################
            hidden_size_prior = 64
            num_layers = 15
            prior = GatedPixelCNN(num_embeddings, hidden_size_prior,
                                  num_layers, n_classes=num_classes).to(device)
            prior.load_state_dict(torch.load(pixelcnn_ckpt))
            prior = prior.to(device)
            prior.eval()
            blackbox = ELBODetector2(k, thresh, encoder, prior, MEAN, STD, log_suffix=log_suffix, log_dir=log_dir)

        else:
            blackbox = VAEDetector(k, thresh, encoder, MEAN, STD, envelope_size=params["envelope_size"],
                                   num_clusters=num_clusters, log_suffix=log_suffix, log_dir=log_dir)
        blackbox.init(blackbox_dir, device, time=created_on, output_type=output_type, T=T)
    else:
        blackbox = Blackbox.from_modeldir(blackbox_dir, device, output_type=output_type, T=T)

    #######################
    # Adaptive encoder
    #######################
    detector_adv = None
    if params["adaptive_adv_ckpt"]:
        print("==> Setting up adaptive encoder...")
        adv_encoder = zoo.get_net(params["adv_encoder_arch_name"], modelfamily, num_classes=num_classes)
        adv_encoder.fc = IdLayer().to(device)
        adv_encoder_ckpt = params["adaptive_adv_ckpt"]
        print(f"=> Loading adv similarity encoder checkpoint '{adv_encoder_ckpt}'")
        checkpoint = torch.load(adv_encoder_ckpt)
        adv_encoder.load_state_dict(checkpoint['state_dict'])
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        adv_encoder.eval()
        
        detector_adv = AdvDetector(k, params["adaptive_thresh"], adv_encoder, MEAN, STD, num_clusters=num_clusters, log_suffix="adv_encoder", log_dir=log_dir)

        detector_adv.init(device)
    else:
        print("==> No adaptive attack")

    # ----------- Initialize Adversary

    # set up query blinding
    blinders_dir = params["blinders_dir"]
    auto_encoder = None
    if blinders_dir is not None:
        blinders_ckp = osp.join(blinders_dir, "checkpoint.blind.pth.tar")
        if osp.isfile(blinders_ckp):
            blinders_noise_fn = blinders_transforms.get_gaussian_noise(device=device, r=0.095)
            auto_encoder = AutoencoderBlinders(blinders_noise_fn)
            print("=> Loading auto-encoder checkpoint '{}'".format(blinders_ckp))
            checkpoint = torch.load(blinders_ckp, map_location=device)
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

    budget = params['budget']
    model_adv_name = params['model_adv']
    model_adv_pretrained = params['pretrained']
    train_epochs = params['train_epochs']
    batch_size = params['batch_size']
    aug_batch_size = params['aug_batch_size']
    eps = params["eps"]
    fb_eps_factor = params["fb_eps_factor"]
    num_steps = params["num_steps"]
    kappa = params['kappa']
    tau = params['tau']
    rho = params['rho']
    take_lastk = params['take_lastk']
    sigma = params['sigma']
    policy = params['policy']
    ema_decay = params['ema_decay']
    binary_search = params['binary_search']
    use_feature_fool = params['featurefool']
    foolbox_alg = params['foolbox_alg']

    random_adv = True if params['random_adv'] else False
    adv_transform = True if params['adv_transform'] else False
    print("Start Extraction with params:")
    print(params)

    # Store arguments
    params_out_path = osp.join(ckp_out_root, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    adversary = JacobianAdversary(blackbox, budget, model_adv_name, model_adv_pretrained, modelfamily, seedset, 
                                  testset, device, ckp_out_root, aug_batch_size=aug_batch_size, num_classes=num_classes, batch_size=batch_size, ema_decay=ema_decay, 
                                  detector=detector_adv, blinder_fn=auto_encoder, binary_search=binary_search, use_feature_fool=use_feature_fool, foolbox_alg=foolbox_alg,
                                  eps=eps, fb_eps_factor=fb_eps_factor, num_steps=num_steps, train_epochs=train_epochs, kappa=kappa, tau=tau, rho=rho, take_lastk=take_lastk,
                                  sigma=sigma, random_adv=random_adv, adv_transform=adv_transform, aug_strategy=policy)


    print('=> constructing transfer set...')
    transferset, model_adv = adversary.get_transferset()
    #import ipdb; ipdb.set_trace()
    #These can be massive (>30G) -- skip it for now
    #transfer_out_path = osp.join(ckp_out_root, 'transferset.pt')

    #torch.save(transferset, transfer_out_path)
    #print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    #torch.save(blackbox.elbo_cumavg, osp.join(ckp_out_root, 'elbo_cumavg.pt'))
    torch.save(blackbox.query_dist, osp.join(ckp_out_root, 'query_dist.pt'))

if __name__ == '__main__':
    main()