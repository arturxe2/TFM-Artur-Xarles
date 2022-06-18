# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:12:45 2022

@author: artur
"""
import pickle
import numpy as np
from tqdm import tqdm

path_store = '/data-local/data3-ssd/axesparraguera'
with open(path_store + '/chunk_list.pkl', 'rb') as f:
    path_list = pickle.load(f)
    
i = 0
for path in tqdm(path_list):
    if i == 0:
        labels = np.load(path + 'labels.npy')
        i += 1
    else:
        labels += np.load(path + 'labels.npy')
        
print(labels)
print(labels / len(path_list) * 100)