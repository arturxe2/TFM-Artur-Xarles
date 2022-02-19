import pandas as pd
import numpy as np
import os
import json

path = '/data-net/datasets/SoccerNetv2/data_split/'
init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'

with open(path + 'train.txt') as f:
    lines = f.readlines()
    
print(init_path + 'england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/')
print(os.listdir(init_path + 'england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/'))

i = 0
chunks = 60
n_total = 0
y_train = ['Elements']
X = []
for line in lines:
    i += 1
    print(line)
    features1 = np.load(init_path + line.rstrip('\n') + '/1_ResNET_TF2.npy')
    features2 = np.load(init_path + line.rstrip('\n') + '/2_ResNET_TF2.npy')
    n_chunks1 = int(features1.shape[0] // chunks)
    n_chunks2 = int(features2.shape[0] // chunks)
    n_total = n_total + n_chunks1 + n_chunks2
    actions = pd.DataFrame(json.load(open(init_path + line.rstrip('\n') + '/Labels-v2.json'))['annotations'])
    actions['half'] = actions['gameTime'].apply(lambda x: int(x[0]))
    actions['minute'] = actions['gameTime'].apply(lambda x: x[4:])
    actions['frame'] = actions['minute'].apply(lambda x: int(x[0:2]) * 60 * 2 + int(x[3:5]) * 2)
    for n in range(n_chunks1):
        x = features1[n * chunks : (n + 1) * chunks, :]
        X.append(x.tolist())
        y = actions['label'][(actions['frame'] >= n * chunks) & (actions['frame'] < (n + 1) * chunks) & (actions['half'] == 1)].values
        if len(y) == 0:
            y = ['']
        else:
            y = y.tolist()
        y_train.append(y)
        

    for n in range(n_chunks2):
        x = features2[n * chunks : (n + 1) * chunks, :]
        X.append(x.tolist())
        y = actions['label'][(actions['frame'] >= n * chunks) & (actions['frame'] < (n + 1) * chunks) & (actions['half'] == 2)].values
        if len(y) == 0:
            y = ['']
        else:
            y = y.tolist()
        y_train.append(y)
    i += 1

    print(i)
    if i == 2:
        break
