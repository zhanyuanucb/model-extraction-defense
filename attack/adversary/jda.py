import random
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import torch.nn.functional as F
import attack.utils.model as model_utils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle
import foolbox
from foolbox.attacks import LinfPGD as PGD
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import Distance


class BaseAugAdversary:
    def __init__(self, adversary_model, blackbox, mean, std, device, blinders_fn=None, log_dir="./"):
        self.adversary_model = adversary_model
        self.blackbox = blackbox
        self.blinders_fn = blinders_fn
        self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1]).to(device)
        self.STD = torch.Tensor(std).reshape([1, 3, 1, 1]).to(device)
        self.device = device
        self.log_path = osp.join(log_dir, "augadv.log")
        self._init_log()

    def _init_log(self):
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as log:
                columns = ["Adv_At", "Label_Diff_At", "Adv_BB", "Label_Diff_BB", "Adv_BB_t", "Label_Diff_BB_t"]
                log.write('\t'.join(columns) + '\n')
        print(f"Created adversary log file at {self.log_path}")

    def _write_log(self, msg):
        with open(self.log_path, 'a') as log:
            log.write('\t'.join(msg) + '\n')

    def normalize(self, images):
        return (images-self.MEAN) / self.STD

    def denormalize(self, images):
        return torch.clamp(images*self.STD + self.MEAN, 0., 1.)   

    def call_blinders(self, images):
        if self.blinders_fn:
            # Query blinding
            with torch.no_grad():
                images = self.denormalize(images)
                images = self.blinders_fn(images)
                images = self.normalize(images)
        return images

    def get_seedset(self, dataloader):
        images, labels = [], []
        for x_t, _ in dataloader:
            x_t = x_t.cuda()
            out = self.blackbox(x_t)
            if isinstance(out, tuple):
                is_adv, y_t = out
            else:
                y_t = out

            y_t = y_t.cpu()
            images.append(x_t.cpu())
            labels.append(y_t)

        seedset = [torch.cat(images), torch.cat(labels)]
        return seedset

    def augment(self, dataloader):
        print("Start data augmentaion...")
        images_aug, labels_aug = [], []
        adv2bb = adv2bb_t = adv2at = 0
        adv_bb2y = adv_bb2y_t = adv_at2y = 0
        is_adv_count = None
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Check victim's and adversary's prediction before augmentation
            with torch.no_grad():
                at_out_before = self.adversary_model(images).argmax(-1)
                bb_out_before = self.blackbox.blackbox(images).argmax(-1)
                yt = labels.argmax(-1)

            # Customize process depending on different augmentation schemes
            process_output = self.process(images, labels)
            if isinstance(process_output, tuple):
                images, is_adv = process_output 
                if idx == 0:
                    is_adv_count = sum(is_adv)
                else:
                    is_adv_count += sum(is_adv)
            else:
                images = process_output

            # Check victim's and adversary's prediction after augmentation
            with torch.no_grad():
                at_out_after = self.adversary_model(images).argmax(-1)
                bb_out_after = self.blackbox.blackbox(images).argmax(-1)
            adv2at += at_out_after.ne(at_out_before).sum().item()
            adv2bb += bb_out_after.ne(bb_out_before).sum().item()
            adv_at2y += at_out_after.ne(yt).sum().item()
            adv_bb2y += bb_out_after.ne(yt).sum().item()

            images_aug.append(images.cpu())

            # Send query to the victim
            images = self.call_blinders(images)
            out = self.blackbox(images)

            if isinstance(out, tuple):
                is_adv, y = out
            else:
                y = out
            adv2bb_t += y.argmax(-1).ne(bb_out_before).sum().item()
            adv_bb2y_t += y.argmax(-1).ne(yt).sum().item()
            labels_aug.append(y.cpu())

        all_images_aug, all_labels_aug = torch.cat(images_aug), torch.cat(labels_aug)

        total_images = all_images_aug.size(0)
        if is_adv_count is not None:
            assert is_adv_count == adv_at2y, f"Isn't consistent to foolbox result: is_adv_count = {is_adv_count}, adv_at2y = {adv_at2y}"
        msg1 = f"{adv_at2y}/{total_images}"
        msg2 = f"{adv2at}/{total_images}"
        msg3 = f"{adv_bb2y}/{total_images}"
        msg4 = f"{adv2bb}/{total_images}"

        print(msg1 + " are adversarial to attacker")
        print(msg2 + " have changed labels after augmentation")
        print(msg3 + " are adversarial to victim before query blinding")
        print(msg4 + " have changed labels after augmentation")
        
        if self.blinders_fn:
            msg5 = f"{adv_bb2y_t}/{total_images}"
            msg6 = f"{adv2bb_t}/{total_images}"
            print(msg5 + " are adversarial to victim after query blinding")
            print(msg6 + " have changed labels after augmentation")
            self._write_log([msg1, msg2, msg3, msg4, msg5, msg6])
        else:    
            self._write_log([msg1, msg2, msg3, msg4, "N/A", "N/A"])
        return all_images_aug, all_labels_aug

    def process(self, images, labels):
        raise NotImplementedError

    def __call__(self, dataloader):
        self.adversary_model.eval()
        return self.augment(dataloader)

######################################
# Jacobian Data Augmentation
######################################
class MultiStepJDA(BaseAugAdversary):
    def __init__(self, adversary_model, blackbox, mean, std, device, 
                 criterion=model_utils.soft_cross_entropy, t_rand=False,
                 blinders_fn=None, eps=0.1, steps=1, momentum=0, delta_step=0, log_dir="./"):
        super(MultiStepJDA, self).__init__(adversary_model, blackbox, mean, std, device, blinders_fn=blinders_fn, log_dir=log_dir)
        self.criterion = criterion
        self.lam = eps/steps
        self.steps = steps
        self.delta_step = delta_step
        self.t_rand = t_rand
        self.momentum = momentum
        self.v = None 

    def reset_v(self, input_shape):
        self.v = torch.zeros(input_shape, dtype=torch.float32).to(self.device)

    def get_jacobian(self, images, labels):
        logits = self.adversary_model(images)
        loss = self.criterion(logits, labels)

        zero_gradients(images)
        loss.backward()
        jacobian = images.grad.data
        return jacobian

    def augment_step(self, images, labels):
        jacobian = self.get_jacobian(images, labels)
        if self.t_rand:
            jacobian *= -1
        self.v = self.momentum * self.v + self.lam*torch.sign(jacobian)
        images = images + self.v
        return images

    def process(self, images, labels):
        if self.t_rand: # If apply t_rand, then augment the images towards some random classes
            targeted_labels = torch.zeros_like(labels)
            batch_size, num_class = labels.size(0), labels.size(1)
            _, cur_labels = torch.topk(labels, 1)
            indices = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                indices[i].append(random.choice([j for j in range(num_class) if j != cur_labels[i]])) 

            indices = torch.Tensor(indices).to(torch.long)
            labels = targeted_labels.scatter(1, indices.to(self.device), torch.ones_like(labels))

        self.reset_v(input_shape=images.shape)
        labels = labels.to(torch.long)

        # Start augmenting images
        for i in range(self.steps):
            images = Variable(images, requires_grad=True, volatile=False)
            images = self.augment_step(images, labels)
        images = images.detach()

        # Clip to valid pixel values
        images = self.denormalize(images)
        images = self.normalize(images)
        return images

   
#################################################
# Adversarial Augmentation
#################################################
class AdvDA(BaseAugAdversary):
    def __init__(self, adversary_model, blackbox, mean, std, device, 
                 blinders_fn=None, eps=0.01, attack_alg=PGD, log_dir="./", **kwarg):
        super(AdvDA, self).__init__(adversary_model, blackbox, mean, std, device, blinders_fn=blinders_fn, log_dir=log_dir)
        self.attack = attack_alg()
        self.get_adv_criterion = Misclassification
        self.eps = eps

    def process(self, images, labels):
        images = images.to(self.device)
        images = self.denormalize(images)
        batch_size = images.size(0)

        labels = labels.argmax(-1).to(self.device)

        # Use foolbox to augment data
        adv_criterion = self.get_adv_criterion(labels)
        fmodel = foolbox.models.PyTorchModel(self.adversary_model, bounds=(0, 1), preprocessing={"mean":self.MEAN, "std":self.STD})
        _, images, is_adv = self.attack(fmodel, images, criterion=adv_criterion, epsilons=self.eps)
        images = self.normalize(images)

        return images, is_adv