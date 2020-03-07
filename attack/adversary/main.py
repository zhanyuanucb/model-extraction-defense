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
from attack.adversary.adv import*
import modelzoo.zoo as zoo
from attack.victim.blackbox import Blackbox

__author__ = "Tribhuvanesh Orekondy"
__author_email__ = "orekondy@mpi-inf.mpg.de"
__adopted_by__ = "Zhanyuan Zhang"
__maintainer__ = "Zhanyuan Zhang"
__maintainer_email__ = "zhang_zhanyuan@berkeley.edu"
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

#def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
#    # Images are either stored as paths, or numpy arrays
#    sample_x = samples[0][0]
#    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))
#
#    if isinstance(sample_x, str):
#        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
#    elif isinstance(sample_x, np.ndarray):
#        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
#    else:
#        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))

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

# ------------ Start
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
else
    print(device)
gpu_count = torch.cuda.device_count()

# ----------- Initialize blackbox
blackbox_dir = params['victim_model_dir']
blackbox = Blackbox.from_modeldir(blackbox_dir, device)

# ----------- Initialize adversary model
model_name = "pnet"
modelfamily = "mnist"
# model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)

if gpu_count > 1:
   model = nn.DataParallel(model)
model = model.to(device)

# ----------- Initialize adversary
nworkers = 10
out_root = "/mydata/model-extraction/model-extraction-defense/models/mnist_testing"
#substitute_out_root = osp.join(out_path, 'substituteset.pickle')
batch_size=128
eps = 0.1
steps=4
momentum=0
adversary = JDAAdversary(model, blackbox, eps=eps, batch_size=batch_size, steps=steps, momentum=momentum)

# ----------- Set up transferset
transferset_path = osp.join(out_root, 'seed.pickle')
with open(transferset_path, 'rb') as rf:
    transferset_samples = pickle.load(rf)
num_classes = transferset_samples[0][1].size(0)
print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

# ----------- Set up testset
dataset_name = "MNIST"
valid_datasets = datasets.__dict__.keys()
modelfamily = datasets.dataset_to_modelfamily[dataset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test']
if dataset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
dataset = datasets.__dict__[dataset_name]
testset = dataset(train=False, transform=transform)
if len(testset.classes) != num_classes:
    raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

# ----------- Set up seed images
np.random.seed(cfg.DEFAULT_SEED)
torch.manual_seed(cfg.DEFAULT_SEED)
torch.cuda.manual_seed(cfg.DEFAULT_SEED)

transferset = TransferSetImagePaths(transferset_samples, transform=transform)
print()
print('=> Training at budget = {}'.format(len(transferset)))

optimizer_name = "adam"
optimizer = get_optimizer(model.parameters(), optimizer_name)
#print(params)

checkpoint_suffix = '.budget{}'.format(budget)
criterion_train = model_utils.soft_cross_entropy

#--------- Extraction
phi = 6
testloader = None
epochs = 100
for p in range(phi):
    if p == phi-1:
        testloader = testset
    print(f"Substitute training epoch {p}")
    print(f"Current size of the substitute set {len(transferset)}")
    _, train_loader = model_utils.train_model(model, transferset, model_dir, epochs=epochs, testset=testloader, criterion_train=criterion_train,
                                              checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer)
                            
    substitute_out_path = osp.join(out_root, f"round-{p}")
    if not osp.exists(substitute_out_dir):
        os.mkdir(substitute_out_dir)
    augset = adversary.JDA(train_loader, substitute_out_dir)
    transferset_samples.extend(augset)
    with open(substitute_out_path, 'wb') as wf:
        pickle.dump(transferset_samples, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset_samples), substitute_out_path))

    transferset = TransferSetImagePaths(transferset_samples, transform=transform)

# Store arguments
#params['created_on'] = str(datetime.now())
#params_out_path = osp.join(model_dir, 'params_train.json')
#with open(params_out_path, 'w') as jf:
#    json.dump(params, jf, indent=True)