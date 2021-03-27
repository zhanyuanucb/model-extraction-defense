# Reference: https://github.com/schoyc/blackbox-detection/blob/master/detection.py

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
import torch.nn.functional as F

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
from attack.adversary.adv import*
import modelzoo.zoo as zoo
from attack.victim.blackbox import Blackbox


class Detector:
    def __init__(self, k, thresh, encoder, mean, std, 
                 num_clusters=50, buffer_size=1000, memory_capacity=100000,
                 log_suffix="", log_dir="./"):
        self.blackbox = None
        self.call_count = 0
        self.detection_count = 0
        self.alarm_count = 0
        self.K = k
        self.thresh = thresh
        self.encoder = encoder
        self.buffer_size = buffer_size
        self.buffer = []
        self.memory = []
        self.MEAN = torch.Tensor(mean).reshape([1, 3, 1, 1])
        self.STD = torch.Tensor(std).reshape([1, 3, 1, 1])
        self.num_clusters = num_clusters
        self.memory_capacity = memory_capacity
        self.memory_size = 0
        self.query_dist = []
        print(f"=> Detector params:\n  k={k}\n  num_cluster={num_clusters}\n thresh={thresh}")

        # Debug
        #self.log_dir = log_dir
        self.log_file = osp.join(log_dir, f"detector.{log_suffix}.log.tsv")
    
    def init(self, blackbox_dir, device, time=None, output_type="one_hot", T=1.):
        self.device = device
        self.MEAN = self.MEAN.to(self.device)
        self.STD = self.STD.to(self.device)
        try:
            print("Load from model dir")
            self.blackbox = Blackbox.from_modeldir(blackbox_dir, device, output_type=output_type, T=T)
        except:
            print("Load from model")
            self.blackbox = Blackbox(blackbox_dir, device, output_type=output_type, T=T)
        self._init_log(time)
    
    def _process(self, images):
        is_adv = [0 for _ in range(images.size(0))]
        with torch.no_grad():
            #images = images * self.STD + self.MEAN
            queries = self.encoder(images).cpu().numpy()
        for i, query in enumerate(queries):
            self.memory_size += 1
            is_attack = self._process_query(query)
            is_adv[i] = 1 if is_attack else 0
        return is_adv

    def _process_query(self, query):
        k = self.K
        if len(self.memory) == 0 and len(self.buffer) < k:
            self.buffer.append(query)
            self.call_count += 1
            return False
        
        all_dists = []

        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)
        
        for queries in self.memory:
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k-1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)
        #print(self.query_count, k_avg_dist)
        self.query_dist.append(k_avg_dist)
        # dist to random queries in history
        #self.query_dist.append(dists[np.random.randint(len(dists))])

        self.buffer.append(query)
        self.call_count += 1

        if len(self.buffer) >= self.buffer_size: # clean buffer
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        is_attack = k_avg_dist < self.thresh
        if is_attack:
            self.detection_count += 1
            if self.detection_count % self.num_clusters == 0:
                self._write_log(k_avg_dist)
                self.alarm_count += 1
                self.clear_memory()
        if self.memory_size >= self.memory_capacity:
            self.clear_memory()
        return is_attack

    def clear_memory(self):
        self.buffer = []
        self.memory = []
        self.memory_size = 0

    def _init_log(self, time):
        if not osp.exists(self.log_file):
            with open(self.log_file, 'w') as log:
                if time is not None:
                    log.write(time + '\n')
                columns = ["Query Count", "Memeory Consumed", "Detection Count", "Detected Distance"]
                log.write('\t'.join(columns) + '\n')
        print(f"Created log file at {self.log_file}")

    def _write_log(self, detected_dist):
        with open(self.log_file, 'a') as log:
            columns = [str(self.call_count), f"{self.memory_size}/{self.memory_capacity}", str(self.detection_count // self.num_clusters), str(detected_dist)]
            log.write('\t'.join(columns) + '\n')
    
    def __call__(self, images, is_adv=False):
        images = images.to(self.device)
        # ---- Going through detection
        is_adv = self._process(images)
        # ----------------------------
        output = self.blackbox(images, is_adv=is_adv)
        return output

    def eval(self):
        self.blackbox.eval()


class VAEDetector(Detector):
    def __init__(self, k, thresh, vae, mean, std, 
                 num_clusters=50, buffer_size=1000, memory_capacity=100000,
                 log_suffix="", log_dir="./"):
        super(VAEDetector, self).__init__(k, thresh, None, mean, std, 
                                          num_clusters=num_clusters, buffer_size=buffer_size, memory_capacity=memory_capacity)
        self.vae = vae
        self.num_samples = 0
        self.pixel_sum = 0.
        self.pixel_sqr_sum = 0.

    def _process(self, images):
        is_adv = [0 for _ in range(images.size(0))]
        B, C, H, W = images.shape
        self.num_samples += B*C*H*W
        with torch.no_grad():
            # Calculate data variance from stream data
            self.pixel_sum += torch.sum(images).item()
            self.pixel_sqr_sum += torch.sum(images**2).item()
            stream_var = self.get_stream_variance()
            self.vae.set_data_variance(stream_var)

            self.call_count += images.size(0)
            #images = images * self.STD + self.MEAN
            nlk = self.vae.get_neglikelihood(images).cpu().numpy().mean()
        self.query_dist.append(nlk)
        return is_adv

    def get_stream_variance(self):
        return (self.pixel_sqr_sum - self.pixel_sum**2/self.num_samples)/self.num_samples

class ELBODetector(VAEDetector):
    def __init__(self, k, thresh, vae, prior, mean, std, 
                 in_channels=1, code_size=8, num_embeddings=512,
                 num_clusters=50, buffer_size=1000, memory_capacity=100000,
                 log_suffix="", log_dir="./"):
        super(ELBODetector, self).__init__(k, thresh, vae, mean, std, 
                                          num_clusters=num_clusters, buffer_size=buffer_size, memory_capacity=memory_capacity)
        self.in_channels = in_channels
        self.code_size = code_size
        self.prior = prior
        self.num_embeddings = num_embeddings

    def get_elbo(self, x):
        z = self.vae._encoder(x)
        z = self.vae._pre_vq_conv(z)
        _, quantized, _, z = self.vae._vq_vae(z)

        z = z.view((-1, self.in_channels, self.code_size, self.code_size)) # (B*CS*CS, 1) -> (B, 1, CS, CS)
        z = z.to(torch.float)
        x_recon = self.vae._decoder(quantized)

        # Log likelihood log(p(x|z))
        llk = -F.mse_loss(x_recon, x) / self.vae.data_variance

        # Log prior log(p(z_q(x)))
        logits = self.prior(z)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.num_embeddings)
        log_prob = F.log_softmax(logits, dim=-1)
        lpz, _ = log_prob.max(dim=-1)
        return llk+lpz

    def _process(self, images):
        is_adv = [0 for _ in range(images.size(0))]
        B, C, H, W = images.shape
        self.num_samples += B*C*H*W
        with torch.no_grad():
            # Calculate data variance from stream data
            self.pixel_sum += torch.sum(images).item()
            self.pixel_sqr_sum += torch.sum(images**2).item()
            stream_var = self.get_stream_variance()
            self.vae.set_data_variance(stream_var)

            self.call_count += images.size(0)
            #images = images * self.STD + self.MEAN
            elbo = self.get_elbo(images).cpu().numpy().mean()
        self.query_dist.append(elbo)
        return is_adv

