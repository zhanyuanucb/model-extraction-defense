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


class Detector:
    def __init__(self, k, thresh, encoder, log_suffix="", log_dir="./"):
        self.blackbox = None
        self.query_count = 0
        self.detection_count = 0
        self.K = k
        self.thresh = thresh
        self.encoder = encoder
        self.detect_adv = False
        self.history = None
        self.log_file = osp.join(log_dir, f"detector.{log_suffix}.log.tsv")
    
    def init(self, blackbox_dir, device):
        self.blackbox = Blackbox.from_modeldir(blackbox_dir, device)
        self._init_log()
    
    def _process_query(self, images):
        encoding = self.encoder(images)
        if self.history is None:
            self.history = encoding.clone()
            return False
        if self.history.size(0) < self.K:
            self.history = torch.cat([self.history, encoding])
            return False
        self.history = torch.cat([self.history, encoding])
        dist_mat = torch.cdist(encoding, self.history)
        dist_mat_k, _ = torch.topk(dist_mat, self.K, largest=False)
        dist_mat_k = dist_mat_k[:, 1:]
        avg_dist_to_k_neighbors = dist_mat_k.mean(dim=-1)
        activated = avg_dist_to_k_neighbors.le(self.thresh).sum().item()
        return activated > 0

    def _init_log(self):
        if not osp.exists(self.log_file):
            with open(self.log_file, 'w') as log:
                columns = ["Query Count", "Detection Count"]
                log.write('\t'.join(columns) + '\n')
        print(f"Created log file at {self.log_file}")

    def _write_log(self):
        with open(self.log_file, 'a') as log:
            columns = [str(self.query_count), str(self.detection_count)]
            log.write('\t'.join(columns) + '\n')
    
    def _reset(self):
        self.history = None
        self.query_count = 0
        self.detect_adv = False

    def __call__(self, images):
        # ---- Going through detection in CPU
        self.query_count += images.size(0)
        self.detect_adv = self._process_query(images)
        if self.detect_adv:
            self.detection_count += 1
            msg = f"{self.query_count} queries: Detected {self.detection_count} adversarial behavior(s)."
            print(msg)
            self._write_log()
            self._reset()
            print("Reset history.")
        # ----------------------------
        images = images.cuda()
        return self.blackbox(images)