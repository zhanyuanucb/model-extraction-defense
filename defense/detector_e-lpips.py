# Reference: https://github.com/schoyc/blackbox-detection/blob/master/detection.py

from collections import defaultdict
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
import kornia

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
from attack.adversary.adv import*
import modelzoo.zoo as zoo
from attack.victim.blackbox import Blackbox

sys.path.append('/mydata/model-extraction/PerceptualSimilarity/models')
import networks_basic2 as lpips_networks

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class ELpipsDetector:
    def __init__(self, k, thresh, num_clusters=50, buffer_size=1000, log_suffix="", log_dir="./",
                       net='alex', lpips_path="/mydata/model-extraction/PerceptualSimilarity/models/weights/v0.1"):
        self.blackbox = None
        self.query_count = 0
        self.detection_count = 0
        self.alarm_count = 0
        self.K = k
        self.thresh = thresh
        self.buffer_size = buffer_size
        self.buffer = defaultdict(list)
        self.memory = defaultdict(list)
        self.num_clusters = num_clusters
        self.log_file = osp.join(log_dir, f"detector.{log_suffix}.log.tsv")

        # pretrained net + linear layer
        self.encoder = lpips_networks.PNetLin(pnet_type=net)
        self.encoder.eval()
        lpips_path = osp.join(lpips_path, f"{net}.pth")
        if osp.isfile(lpips_path):
            print('Loading model from: %s'%lpips_path)
            self.encoder.load_state_dict(torch.load(lpips_path), strict=False)
        else:
            print("Can't find weights in: %s"%lpips_path)
            exit(1)
        self.lins = self.encoder.lins
        self.L = self.encoder.L
    
    def init(self, blackbox_dir, device, time=None):
        self.device = device
        self.encoder = self.encoder.to(self.device)
        #self.encoder = self.encoder
        self.blackbox = Blackbox.from_modeldir(blackbox_dir, device)
        self._init_log(time)
    
    def _process(self, images):
        k = self.K
        batch_size, c, h, w = images.shape
        is_adv = [0 for _ in range(batch_size)]
        with torch.no_grad():
            if h != 224:
                images = kornia.resize(images, (224, 224))
            queries = self.encoder(images)

        for i in range(batch_size):
            self.query_count += 1
            if len(self.memory[0]) == 0 and len(self.buffer[0]) < k:
                for kk in range(self.L):
                    query = queries[kk][i]
                    self.buffer[kk].append(query)
                continue

            res = []
            for kk in range(self.L):
                query = queries[kk][i]
                res.append(self._process_layer(query, kk))
            val = res[0]
            for l in range(1, self.L):
                val += res[l]
            val = res[0]
            for l in range(1,self.L):
                val += res[l]

            val = val.cpu().numpy()
            #val = val.numpy()
            k_nearest_dists = np.partition(val, k-1)[:k, None]
            k_avg_dist = np.mean(k_nearest_dists)

            is_attack = k_avg_dist < self.thresh
            if is_attack:
                self.detection_count += 1
                if self.detection_count % self.num_clusters == 0:
                    self._write_log(k_avg_dist)
                    self.alarm_count += 1
                    self.clear_memory()

            is_adv[i] = 1 if is_attack else 0
        return is_adv

    def _process_layer(self, query, kk):
        k = self.K
        
        layer_res = []

        if len(self.buffer[kk]) > 0:
            queries = torch.stack(self.buffer[kk], dim=0)
            diff = (queries - query)**2 
            with torch.no_grad():
                layer_res.append(spatial_average(self.lins[kk].model(diff), keepdim=True))
        
        for queries in self.memory[kk]:
            diff = (queries - query)**2 
            with torch.no_grad():
                layer_res.append(spatial_average(self.lins[kk].model(diff), keepdim=True)) 

        layer_res = torch.cat(layer_res)

        self.buffer[kk].append(query)
        if len(self.buffer[kk]) >= self.buffer_size: # clean buffer
            self.memory[kk].append(torch.stack(self.buffer[kk], dim=0))
            self.buffer[kk] = []

        return layer_res

    def clear_memory(self):
        self.buffer = defaultdict(list)
        self.memory = defaultdict(list)

    def _init_log(self, time):
        if not osp.exists(self.log_file):
            with open(self.log_file, 'w') as log:
                if time is not None:
                    log.write(time + '\n')
                columns = ["Query Count", "Detection Count", "Detected Distance"]
                log.write('\t'.join(columns) + '\n')
        print(f"Created log file at {self.log_file}")

    def _write_log(self, detected_dist):
        with open(self.log_file, 'a') as log:
            columns = [str(self.query_count), str(self.detection_count // self.num_clusters), str(detected_dist)]
            log.write('\t'.join(columns) + '\n')
    
    def __call__(self, images):
        # ---- Going through detection
        images = images.to(self.device)
        is_adv = self._process(images)
        #images = images.to(self.device)
        # ----------------------------
        output = self.blackbox(images)
        return is_adv, output