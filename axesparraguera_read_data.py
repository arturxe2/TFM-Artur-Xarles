import pandas as pd
import numpy as np

path = '/data-net/datasets/SoccerNetv2/data_split/'
init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'

with open(path + 'train.txt') as f:
    lines = f.readlines()

i = 0
for line in lines:
    i += 1
    print(line)
    features = np.load(init_path + line + '1_ResNet_TF2.npy')
    print(features.shape)
    if i == 2:
        break
