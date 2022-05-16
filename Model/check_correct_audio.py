# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:23:20 2022

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

splits = ['train', 'valid', 'test', 'challenge']
path="/data-local/data1-hdd/axesparraguera/vggish"
features="audio.npy"

for split in splits:
    j = 0
    s = 0
    for game in getListGames(split):
        
        s += 1
  
        #Check if exists audio.npy
        exists_audio = os.path.exists(os.path.join(path, game, "1_" + features))
    
        if exists_audio:
        # Load features
            feat_half1 = torch.from_numpy(np.load(os.path.join(path, game, "1_" + features))).cuda()
            feat_half2 = torch.from_numpy(np.load(os.path.join(path, game, "2_" + features))).cuda()
            print(feat_half1.shape)
            print(feat_half2.shape)
        
        if (feat_half1.shape[0] > 100) & (feat_half2.shape[0] > 100):
            j += 1
            
    print('Split: ' + split + '. \nMatches with correct audio: ' + str(j) + '. \nMatches with incorrect audio: ' + str(s - j))
