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
import math


from tqdm import tqdm
# import utils

import torch

import logging
import json
from SoccerNet.Downloader import getListGames

model_name = 'final_model'
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
path="/data-local/data1-hdd/axesparraguera/vggish"
features="audio.npy"

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.embeddings.register_forward_hook(get_activation('embeddings'))
for game in tqdm(listGames):
  
    #Check if exists audio.npy
    exists_audio = os.path.exists(os.path.join(path, game, "1_" + features))
    
    if exists_audio:
        # Load features
        feat_half1 = torch.from_numpy(np.load(os.path.join(path, game, "1_" + features))).cuda()
        feat_half2 = torch.from_numpy(np.load(os.path.join(path, game, "2_" + features))).cuda()
        print(feat_half1.shape)
        print(feat_half2.shape)
        
        if (feat_half1.shape[0] > 100) & (feat_half2.shape[0] > 100):
            embed_half1 = []
            embed_half2 = []
            
            activation = {}
            for j in range(0, math.ceil((feat_half1.shape[0] - 1) // 100) + 1):
                output = model(feat_half1[(j * 100): np.minimum((j+1) * 100, feat_half1.shape[0]), :, :])
                embed_half1.append(activation['embeddings'].cpu().numpy())
            for j in range(0, math.ceil((feat_half2.shape[0] - 1) // 100) + 1):
                output = model(feat_half2[(j * 100): np.minimum((j+1) * 100, feat_half2.shape[0]), :, :])
                embed_half2.append(activation['embeddings'].cpu().numpy())
                
            
                
            embed_half1 = np.concatenate(embed_half1)
            embed_half2 = np.concatenate(embed_half2)
            
            n1 = len(embed_half1)
            n2 = len(embed_half2)
            embed_half1 = embed_half1[np.delete(np.arange(0, n1), np.arange(12, n1, 25)), :]
            embed_half2 = embed_half2[np.delete(np.arange(0, n2), np.arange(12, n2, 25)), :]
            
        else:
            aux1 = embed_half1.mean(axis = 0)
            embed_half1 = np.array([aux1] * 2700)
            aux2 = embed_half2.mean(axis = 0)
            embed_half2 = np.array([aux2] * 2700)
    
    else:
        aux1 = embed_half1.mean(axis = 0)
        embed_half1 = np.array([aux1] * 2700)
        aux2 = embed_half2.mean(axis = 0)
        embed_half2 = np.array([aux2] * 2700)
    
    print(embed_half1.shape)
    print(embed_half2.shape)    
    
    np.save(os.path.join(path, game, "1_featA2.npy"), embed_half1)
    np.save(os.path.join(path, game, "2_featA2.npy"), embed_half2)
    
