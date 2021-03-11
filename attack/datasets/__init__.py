from torchvision import transforms
from torch import nn
import kornia.augmentation as kaug

#from knockoff.datasets.diabetic5 import Diabetic5
import sys
sys.path.append('/mydata/model-extraction/model-extraction-defense/')
from attack.datasets.caltech256 import Caltech256
from attack.datasets.cubs200 import CUBS200
from attack.datasets.cifarlike import CIFAR10, CIFAR100, SVHN
from attack.datasets.imagenet1k import ImageNet1k
from attack.datasets.imagenet32 import ImageNet32
from attack.datasets.cinic10 import CINIC10
from attack.datasets.indoor67 import Indoor67
from attack.datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from attack.datasets.tinyimagenet200 import TinyImageNet200

# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_modelfamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'EMNIST': 'mnist',
    'EMNISTLetters': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'CIFAR10': 'cifar',
    'CIFAR100': 'cifar',
    'SVHN': 'cifar',
    #'TinyImageNet200': 'cifar',
    'TinyImageNet200': 'cifar_jb',

    # Imagenet
    #'CUBS200': 'imagenet',
    #'Caltech256': 'imagenet',
    #'Indoor67': 'imagenet',
    #'Diabetic5': 'imagenet',
    #'ImageNet1k': 'imagenet',

    'CUBS200': 'imagenet_jb',
    'Caltech256': 'imagenet_jb',
    'Indoor67': 'imagenet_jb',
    'Diabetic5': 'imagenet_jb',
    'ImageNet1k': 'imagenet_jb',

    # CINIC10
    'CINIC10': "cinic10"
}

# Transforms
modelfamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(32),
                                     transforms.RandomRotation(60),
                                     transforms.RandomAffine(0, translate=(0.45, 0.45)),
                                     transforms.ColorJitter(brightness=0.5),
                                     transforms.ColorJitter(contrast=0.55)
                                     ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'train2': transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(32),
#                                     transforms.RandomRotation(45),
#                                     transforms.RandomAffine(0, translate=(0.45, 0.45)),
#                                     transforms.ColorJitter(brightness=0.5),
#                                     transforms.ColorJitter(contrast=0.55)
                                     ]),
            transforms.ToTensor()
        ]),

        'test2': transforms.ToTensor(),

    },
    'cifar_jb': {
        'train': transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'test': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),

    },

    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    },

    'imagenet_jb': {
        'train': transforms.Compose([
            transforms.Resize(37),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(37),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'cinic10': {
        'train': transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(32),
                                     transforms.RandomRotation(45),
                                     transforms.RandomAffine(0, translate=(0.45, 0.45)),
                                     transforms.ColorJitter(brightness=0.5),
                                     transforms.ColorJitter(contrast=0.55)
                                     ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                 std=[0.24205776, 0.23828046, 0.25874835])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                 std=[0.24205776, 0.23828046, 0.25874835])
        ])
    }
}
