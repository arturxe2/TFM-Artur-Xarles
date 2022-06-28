'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file analyzes the different actions occurance in the dataset
'''
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time
import pickle
import blosc


from tqdm import tqdm
# import utils

import json
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
listGames = getListGames(['train', 'test', 'valid'])
    
i = 0
label_count = np.zeros(17)
for game in tqdm(listGames):
    labels = json.load(open(os.path.join(path_labels, game, "Labels-v2.json")))
    
    
    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        if event not in EVENT_DICTIONARY_V2:
            continue
        label = EVENT_DICTIONARY_V2[event]
        label_count[label] += 1


print(label_count)
print(EVENT_DICTIONARY_V2)