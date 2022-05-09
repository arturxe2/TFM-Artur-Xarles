# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:07:00 2022

@author: artur
"""
from __future__ import print_function
from SoccerNet.Evaluation.utils import AverageMeter
import time

from random import shuffle

import numpy as np


#import vggish_input
#import vggish_params
#import vggish_slim
from vggish_torch import *
import torch.nn as nn

from torch.utils.data import Dataset

import random
# import pandas as pd
import os



from tqdm import tqdm
# import utils

import torch

import logging
import json
from SoccerNet.Downloader import getListGames

model_name = 'model'
model = VGGish(urls = model_urls, pretrained = True, preprocess = False, postprocess=False)
checkpoint = torch.load(os.path.join("models", model_name, "model.pth.tar"))
model.load_state_dict(checkpoint['state_dict'])
'''
model.classifier = Postprocessor()
state_dict = hub.load_state_dict_from_url(model_urls['pca'], progress=True)
    # TODO: Convert the state_dict to torch
state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
)
state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
)
model.classifier.load_state_dict(state_dict)
'''
model = model.cuda()


listGames = getListGames(['train', 'valid', 'test', 'challenge'])
path="/data-net/datasets/SoccerNetv2/videos_lowres"
features="audio.npy"

for game in tqdm(listGames):
  

    # Load features
    feat_half1 = torch.from_numpy(np.load(os.path.join(path, game, "1_" + features))).cuda()
    feat_half2 = torch.from_numpy(np.load(os.path.join(path, game, "2_" + features))).cuda()

    print(model(feat_half1[0:3, :, :]))