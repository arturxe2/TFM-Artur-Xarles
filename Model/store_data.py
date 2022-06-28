'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file uses SoccerNetClipsTrain in dataset.py to store samples to train the HMTAS model
'''
from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

stored = SoccerNetClipsTrain(store = True, chunk_size = 3, stride = 1, split = ['train'])
print(stored.__len__())
print(stored.__getitem__(0))

