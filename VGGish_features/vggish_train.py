
r"""A simple demonstration of running VGGish in training mode.
This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.
For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.
Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100
  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

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
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from loss import NLLLoss_weights




class TrainVGGish(Dataset):
    def __init__(self, path="/data-net/datasets/SoccerNetv2/videos_lowres", features="audio.npy", labels="labels.npy", 
                 split=["train", "valid"], version=2, val_split = 0.8):
        self.path = path
        self.features = features
        self.labels = labels
        self.listGames = getListGames(split)
        self.version = version
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17

        logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)


        logging.info("Read examples")
        
        self.game_feats = list()
        self.game_labels = list()
        i = 0
        for game in tqdm(self.listGames):
            i += 1
            if i < 10:

                # Load features
                feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
                feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
                labels_half1 = np.load(os.path.join(self.path, game, "1_" + self.labels))
                labels_half2 = np.load(os.path.join(self.path, game, "2_" + self.labels))
        
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(labels_half1)
                self.game_labels.append(labels_half2)
                
                #except:
                    #print('Not npy file')
                
        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)

        self.n = self.game_feats.shape[0]
        
        
    def __getitem__(self, index):
        return self.game_feats[index, :, :], self.game_lables[index, :]
    
    def __len__(self):
        return self.n

        
        



if __name__ == '__main__':

    model = get_vggish(with_classifier=True, pretrained=True)
    model.classifier._modules['2'] = nn.Linear(100, 18)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, 
                                betas=(0.9, 0.999), eps=1e-08, 
                                weight_decay=1e-5, amsgrad=True)
    criterion = NLLLoss_weights()
    dataset_Train = TrainVGGish()
    train_loader = torch.utils.data.DataLoader(dataset_Train,
        batch_size=128, pin_memory=True)
    print(model)
    