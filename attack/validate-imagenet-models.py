import numpy as np
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name(0))

#resnet50_target = models.resnet50(pretrained=True).to(device)
#resnet50_target.eval()
resnet50_adv = models.resnet50().to(device)
resnet50_adv.eval()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          normalize])

# TODO: change directory
imagenet_val = datasets.ImageNet('/data/imagenet', split='val', download=False, 
                                 transform=imagenet_transform)

np.random.seed(42)
N = 10000
val_subset_indices = np.random.choice(np.arange(50000), size=N, replace=False)
val_loader = torch.utils.data.DataLoader(imagenet_val, 
                                         batch_size=128,
                                         num_workers=4,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))
                                        
def validate(val_loader, model, device, k=5):
    """Validate model's top-k accuracy
    Input:
    - val_loader: pytorch data loader.
    - model: pytorch model
    - device: context
    - k (int): top-k accuracy
    - clip (function): If not None, apply clipping on patch logits
    Return:
    val_acc: float
    """
    # switch to evaluate mode
    model.eval()
    total_iter = len(val_loader)
    cum_acc = 0
    val_time = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            logits = model(images)
            # measure accuracy
            _, y_hat = torch.topk(logits, k=k, dim=1)
            y_hat, target = y_hat.cpu().numpy(), target.cpu().numpy()
            acc = sum([target[i] in y_hat[i] for i in range(target.size)]) / target.size
            cum_acc += acc
    val_acc = cum_acc / total_iter
    msg = 'Validation accuracy: {:.3f}'.format(val_acc)
    print(msg)
    return val_acc


tic = time.time()
validate(val_loader, resnet50_target, device)
tac = time.time()
print(f"time: {(tac - tic)/60} mins")

tic = time.time()
validate(val_loader, resnet50_adv, device)
tac = time.time()
print(f"time: {(tac - tic)/60} mins")