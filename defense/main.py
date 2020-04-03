import torchvision.models as models
import json
import os
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
#from attack.adversary.adv import*
import modelzoo.zoo as zoo
#from attack.victim.blackbox import Blackbox
from detector import *

__author = "Zhanyuan Zhang"
__author_email__ = "zhang_zhanyuan@berkeley.edu"
__reference__ = "https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/train.py"
__status__ = "Development"

#----------- Helper functions
class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

class MNISTSeedsetImagePaths(ImageFolder):
    """MNIST Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image = image[0][None] # only use the first channel
        return image, target


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class ImagePicklePath(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        pickle_path, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        with open(pickle_path, 'rb') as file:
            img = pickle.load(file)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

params = {"model_name":"resnet18",
          "modelfamily":"cifar",
          "num_classes":10,
          "out_root":"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/",
          "batch_size":128,
          "eps":0.1,
          "steps":4,
          "phi":3,
          "epochs":10,
          "momentum":0,
          "blackbox_dir":'/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wo_normalization',
          "seedset_dir":"/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10",
          "testset_name":"CIFAR10",
          "optimizer_name":"adam",
          "encoder_ckp":"/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/margin-3.2",
          "encoder_margin":3.2,
          "k":200,
          "thresh":0.334,
          "log_suffix":"testing",
          "log_dir":"./"}

# ------------ Start
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    print(device)
gpu_count = torch.cuda.device_count()

# ----------- Initialize detector
k = params["k"]
thresh = params["thresh"]
log_suffix = params["log_suffix"]
log_dir = params["log_dir"]
modelfamily = params["modelfamily"]
num_classes = 10
encoder = zoo.get_net("simnet", modelfamily, num_classes=num_classes)

#             Setup encoder
encoder_ckp = params["encoder_ckp"]
encoder_margin = params["encoder_margin"]
ckp = osp.join(encoder_ckp, f"checkpoint.sim-{encoder_margin}.pth.tar")
print(f"=> loading encoder checkpoint '{ckp}'")
checkpoint = torch.load(ckp)
start_epoch = checkpoint['epoch']
encoder.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

#encoder = encoder.to(device)

detector = Detector(k, thresh, encoder, log_suffix=log_suffix, log_dir=log_dir)
blackbox_dir = params["blackbox_dir"]
detector.init(blackbox_dir, device)

# ----------- Initialize adversary model
model_name = params["model_name"]
# model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

if gpu_count > 1:
   model = nn.DataParallel(model)
model = model.to(device)

# ----------- Initialize adversary
nworkers = 10
out_root = params["out_root"]
#substitute_out_root = osp.join(out_path, 'substituteset.pickle')
batch_size = params["batch_size"]
eps = params["eps"]
steps= params["steps"]
momentum= params["momentum"]
adversary = JDAAdversary(model, detector, eps=eps, batch_size=batch_size, steps=steps, momentum=momentum)

# ----------- Set up transferset
seedset_path = osp.join(params["seedset_dir"], 'seed.pickle')
with open(seedset_path, 'rb') as rf:
    transferset_samples = pickle.load(rf)
num_classes = transferset_samples[0][1].size(0)
print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

# ----------- Set up testset
testset_name = params["testset_name"]
valid_datasets = datasets.__dict__.keys()
modelfamily = datasets.dataset_to_modelfamily[testset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test2'] # test2 has no normalization
if testset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
dataset = datasets.__dict__[testset_name]
testset = dataset(train=False, transform=transform)
if len(testset.classes) != num_classes:
    raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

# ----------- Set up seed images
np.random.seed(cfg.DEFAULT_SEED)
torch.manual_seed(cfg.DEFAULT_SEED)
torch.cuda.manual_seed(cfg.DEFAULT_SEED)

transferset = ImagePicklePath(transferset_samples)
print()
print('=> Training at budget = {}'.format(len(transferset)))

optimizer_name = params["optimizer_name"]
optimizer = get_optimizer(model.parameters(), optimizer_name)
#print(params)

criterion_train = model_utils.soft_cross_entropy

#--------- Extraction
phi = params["phi"]
steps = params["steps"]
budget = (steps+1)**phi*len(transferset)
checkpoint_suffix = 'budget{}'.format(budget)
testloader = testset
epochs = params["epochs"]
num_workers = 10
train_loader = DataLoader(transferset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
for p in range(1, phi+1):
    substitute_out_dir = osp.join(out_root, f"round-{p}")
    if not osp.exists(substitute_out_dir):
        os.mkdir(substitute_out_dir)
    augset = adversary.JDA(train_loader, substitute_out_dir)
    transferset_samples.extend(augset)
    substitute_out_path = osp.join(out_root, f"substitute_set{p}.pickle")
    with open(substitute_out_path, 'wb') as wf:
        pickle.dump(transferset_samples, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset_samples), substitute_out_path))

    transferset = ImagePicklePath(transferset_samples)
    print(f"Substitute training epoch {p}")
    print(f"Current size of the substitute set {len(transferset)}")
    _, train_loader = model_utils.train_model(model, transferset, out_root, epochs=epochs, testset=testloader, criterion_train=criterion_train,
                                              checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)
                            
# Store arguments
params['budget'] = budget
params['created_on'] = str(datetime.now())
params_out_path = osp.join(out_root, 'params_train.json')
with open(params_out_path, 'w') as jf:
    json.dump(params, jf, indent=True)