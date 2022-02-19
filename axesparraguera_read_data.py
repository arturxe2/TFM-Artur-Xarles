import pandas as pd
import numpy as np
import os

path = '/data-net/datasets/SoccerNetv2/data_split/'
init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'

with open(path + 'train.txt') as f:
    lines = f.readlines()
    
print(init_path + 'england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/')
print(os.listdir(init_path + 'england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/'))

i = 0
for line in lines:
    i += 1
    print(line)
    features = np.load(init_path + line.rstrip('\n') + '/')
    print(features.shape)
    if i == 2:
        break
