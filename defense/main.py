import torchvision.models as models
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
sys.path.append('/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding')
import os.path as osp
import pickle
from datetime import datetime
import torch

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

## means need to change for different dataset
params = {"model_name":"resnet18", ##
          "num_classes":10,
          "out_root":"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/", ##
          "batch_size":32,
          "eps":0.01,
          "steps":5,
          "phi":3, # Budget = (steps+1)**phi*len(transferset)
          "alt_t": None, # Alternate period of step size sign
          "epochs":10, 
          "momentum":0,
          "blackbox_dir":'/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28', ##
          "blinders_dir":'/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_', ##
          #"blinders_dir":None, ##
          "seedset_dir":"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10", ##
          "testset_name":"CIFAR10", ##
          "optimizer_name":"adam",
          "use_detector": True,
          "encoder_arch_name": "simnet",
          "encoder_ckp":"/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/",
          "encoder_margin":3.2,
          "k":10,
          "thresh":0.0397684188708663, ##
          "log_suffix":"testing",
          "log_dir":"./"}

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
if blackbox_dir is not None:
    encoder_ckp = params["encoder_ckp"]
    encoder_margin = params["encoder_margin"]
    encoder_ckp = osp.join(encoder_ckp, encoder_arch_name, f"{testset_name}-margin-{encoder_margin}")
    ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
    print(f"=> loading encoder checkpoint '{ckp}'")
    checkpoint = torch.load(ckp)
    start_epoch = checkpoint['epoch']
    encoder.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

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
out_root = params["out_root"]

ckp_out_root = osp.join(out_root, created_on)
if not osp.exists(ckp_out_root):
    os.mkdir(ckp_out_root)

# attack parameters
eps = params["eps"]
steps= params["steps"]
momentum= params["momentum"]

# set up query blinding
blinders_dir = params["blinders_dir"]
if blinders_dir is not None:
    blinders_noise_fn = blinders_transforms.get_gaussian_noise(device=device, sigma=0.095)
    auto_encoder = AutoencoderBlinders(blinders_noise_fn)
    blinders_ckp = osp.join(blinders_dir, "checkpoint.blind.pth.tar")
    if osp.isfile(blinders_ckp):
        print("=> Loading auto-encoder checkpoint '{}'".format(blinders_ckp))
        checkpoint = torch.load(blinders_ckp)
        start_epoch = checkpoint['epoch']
        best_test_loss = checkpoint['best_loss']
        auto_encoder.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print(f"=> Best val loss: {best_test_loss}")
    else:
        print("=> no checkpoint found at '{}'".format(blinders_ckp))
        exit(1)
    auto_encoder = auto_encoder.to(device)
    auto_encoder.eval()
else:
    auto_encoder = None

adversary = MultiStepJDA(model, blackbox, MEAN, STD, device, blinders_fn=auto_encoder, eps=eps, steps=steps, momentum=momentum) 

# ----------- Set up seedset
seedset_path = osp.join(params["seedset_dir"], 'seed.pt')
images_sub, labels_sub = torch.load(seedset_path)
seedset_samples = [images_sub, labels_sub]
num_classes = seedset_samples[1][0].size(0)
print('=> found transfer set with {} samples, {} classes'.format(seedset_samples[0].size(0), num_classes))

# ----------- Set up testset
valid_datasets = datasets.__dict__.keys()
test_transform = datasets.modelfamily_to_transforms[modelfamily]['test'] # test2 has no normalization
if testset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
dataset = datasets.__dict__[testset_name]
testset = dataset(train=False, transform=test_transform)

if len(testset.classes) != num_classes:
    raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

# ----------- Set up seed images
substitute_set = ImageTensorSet(seedset_samples)
print('=> Training at budget = {}'.format(len(substitute_set)))

optimizer_name = params["optimizer_name"]
optimizer = model_utils.get_optimizer(model.parameters(), optimizer_name)

criterion_train = model_utils.soft_cross_entropy

#--------- Extraction
phi = params["phi"]
alt_t = params["alt_t"]
steps = params["steps"]
batch_size = params["batch_size"]
#budget = (steps+1)**phi*len(substitute_set)
budget = (phi+1)*len(substitute_set)
checkpoint_suffix = 'budget{}'.format(budget)
testloader = testset
epochs = params["epochs"]
num_workers = 10
train_loader = DataLoader(substitute_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
aug_loader = DataLoader(substitute_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
substitute_out_path = osp.join(out_root, f"substitute_set.pt")
for p in range(1, phi+1):
    if alt_t: # Apply periodic step size
        adversary.JDA.lam *= (-1)**(p//alt_t)
    images_aug, labels_aug = adversary(aug_loader)
    nxt_aug_samples = [images_aug.clone(), labels_aug.clone()]
    nxt_aug_set = ImageTensorSet(nxt_aug_samples)
    aug_loader = DataLoader(nxt_aug_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    images_sub = torch.cat([images_sub, images_aug.clone()])
    labels_sub = torch.cat([labels_sub, labels_aug.clone()])
    substitute_samples = [images_sub, labels_sub]
    torch.save(substitute_samples, substitute_out_path)
    print('=> substitute set ({} samples) written to: {}'.format(substitute_samples[0].size(0), substitute_out_path))

    substitute_set = ImageTensorSet(substitute_samples)
    print(f"Substitute training epoch {p}")
    print(f"Current size of the substitute set {len(substitute_set)}")
    _, train_loader = model_utils.train_model(model, substitute_set, ckp_out_root, batch_size=batch_size, epochs=epochs, testset=testloader, criterion_train=criterion_train,
                                              checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)
                            
# Store arguments
params['budget'] = images_sub.size(0)
params['num_pruned'] = budget-params['budget']
params_out_path = osp.join(ckp_out_root, 'params_train.json')
with open(params_out_path, 'w') as jf:
    json.dump(params, jf, indent=True)