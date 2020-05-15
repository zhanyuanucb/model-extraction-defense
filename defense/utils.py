from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from PIL import Image

class ImageTensorSet(Dataset):
    """
    Data are saved as:
    List[data:torch.Tensor(), labels:torch.Tensor()]
    """
    def __init__(self, samples, transform=None, dataset="CIFAR10"):
        self.data, self.targets = samples
        self.transform = transform
        self.mode = "RGB" if dataset != "MNIST" else "L"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img.numpy().transpose([1, 2, 0]), mode=self.mode)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

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
