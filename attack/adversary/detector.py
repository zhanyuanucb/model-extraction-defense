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

import attack.config as cfg
import attack.utils.model as model_utils
from attack import datasets
from attack.adversary.adv import*
import modelzoo.zoo as zoo
from attack.victim.blackbox import Blackbox


class AdvDetector:
    def __init__(self, k, thresh, encoder, mean, std, 
                 num_clusters=50, buffer_size=1000, memory_capacity=10000,
                 log_suffix="", log_dir="./"):
        #self.blackbox = None
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

        # Debug
        #self.log_dir = log_dir
        self.log_file = osp.join(log_dir, f"detector.{log_suffix}.log.tsv")
    
    def init(self, blackbox_dir, device, time=None, output_type="one_hot", T=1.):
        self.device = device
        self.MEAN = self.MEAN.to(self.device)
        self.STD = self.STD.to(self.device)
        #self.blackbox = Blackbox.from_modeldir(blackbox_dir, device, output_type=output_type, T=T)
        #self._init_log(time)
    
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

#    def _init_log(self, time):
#        if not osp.exists(self.log_file):
#            with open(self.log_file, 'w') as log:
#                if time is not None:
#                    log.write(time + '\n')
#                columns = ["Query Count", "Memeory Consumed", "Detection Count", "Detected Distance"]
#                log.write('\t'.join(columns) + '\n')
#        print(f"Created log file at {self.log_file}")
#
#    def _write_log(self, detected_dist):
#        with open(self.log_file, 'a') as log:
#            columns = [str(self.call_count), f"{self.memory_size}/{self.memory_capacity}", str(self.detection_count // self.num_clusters), str(detected_dist)]
#            log.write('\t'.join(columns) + '\n')
    
    def __call__(self, images):
        images = images.to(self.device)
        # ---- Going through detection
        is_adv = self._process(images)
        # ----------------------------
        return is_adv