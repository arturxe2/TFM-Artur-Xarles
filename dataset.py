from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json
import random
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

def mix_up(feat1, feat2, y1, y2):
    lam = np.random.beta(a = 0.2, b=0.2, size=1)
    feat_new = feat1 * lam + feat2 * (1-lam)
    y_new = y1 * lam + y2 * (1-lam)
    feat_new_list = [l.tolist() for l in feat_new]
    y_new_list = (y_new).tolist()
    return feat_new_list, y_new_list



def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
    # for i in torch.arange(0, clip_length):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
        # Not replicate last, but take the clip closest to the end of the video
        # idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length
    # print(idx)
    return feats[idx,...]

# def feats2clip(feats, stride, clip_length, padding = "replicate_last"):
#     if padding =="zeropad":
#         print("beforepadding", feats.shape)
#         pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
#         print("pad need to be", clip_length-pad)
#         m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
#         feats = m(feats)
#         print("afterpadding", feats.shape)
#         # nn.ZeroPad2d(2)

#     idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
#     idxs = []
#     for i in torch.arange(0, clip_length):
#         idxs.append(idx+i)
#     idx = torch.stack(idxs, dim=1)

#     if padding=="replicate_last":
#         idx = idx.clamp(0, feats.shape[0]-1)
#         # Not replicate last, but take the clip closest to the end of the video
#         idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

#     return feats[idx,:]

class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2, chunk_size=240, augment = False):
        self.path = path
        labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.version = version
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)


        logging.info("Pre-compute clips")
        
        if self.path != 'Baidu+ResNet':
            self.game_feats = list()
        else:
            self.game_feats1 = list()
            self.game_feats2 = list()
        self.game_labels = list()

        # game_counter = 0
        baidu_path = '/data-net/datasets/SoccerNetv2/Baidu_features'
        baidu_name = 'baidu_soccer_embeddings.npy'
        resnet_path = '/home-net/axesparraguera/data/VGGFeatures'
        resnet_name = 'VGGish.npy'
        stride = self.chunk_size #// 2
        for game in tqdm(self.listGames):
            # Load features
            if self.path != 'Baidu+ResNet':
                feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
                feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
                feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
                feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])
                feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=stride, clip_length=self.chunk_size)            
                feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=stride, clip_length=self.chunk_size)
            # print("feat_half1.shape",feat_half1.shape)
            else:
                feat_half1B = np.load(os.path.join(baidu_path, game, "1_" + baidu_name))
                feat_half1B = feat_half1B.reshape(-1, feat_half1B.shape[-1])
                feat_half1R = np.load(os.path.join(resnet_path, game, "1_" + resnet_name))
                feat_half1R = feat_half1R.reshape(-1, feat_half1R.shape[-1])
                feat_half2B = np.load(os.path.join(baidu_path, game, "2_" + baidu_name))
                feat_half2B = feat_half2B.reshape(-1, feat_half2B.shape[-1])
                feat_half2R = np.load(os.path.join(resnet_path, game, "2_" + resnet_name))
                feat_half2R = feat_half2R.reshape(-1, feat_half2R.shape[-1])
                
                if feat_half1B.shape[0]*2 > feat_half1R.shape[0]:
                    print('Different shape')
                    print('Previous shape: ' + str(feat_half1R.shape))
                    feat_half1R_aux = np.zeros((feat_half1B.shape[0] * 2, feat_half1R.shape[1]))
                    feat_half1R_aux[:feat_half1R.shape[0]] = feat_half1R
                    feat_half1R_aux[feat_half1R.shape[0]:] = feat_half1R[feat_half1R.shape[0]-1]
                    feat_half1R = feat_half1R_aux
                    print('Resized to: ' + str(feat_half1R.shape))
                    
                if feat_half2B.shape[0]*2 > feat_half2R.shape[0]:
                    print('Different shape')
                    print('Previous shape: ' + str(feat_half2R.shape))
                    feat_half2R_aux = np.zeros((feat_half2B.shape[0] * 2, feat_half2R.shape[1]))
                    feat_half2R_aux[:feat_half2R.shape[0]] = feat_half2R
                    feat_half2R_aux[feat_half2R.shape[0]:] = feat_half2R[feat_half2R.shape[0]-1]
                    feat_half2R = feat_half2R_aux
                    print('Resized to: ' + str(feat_half2R.shape))
                    
                if feat_half1B.shape[0]*2 < feat_half1R.shape[0]:
                    print('Different shape')
                    print('Previous shape: ' + str(feat_half1B.shape))
                    feat_half1B_aux = np.zeros((feat_half1R.shape[0] // 2, feat_half1B.shape[1]))
                    feat_half1B_aux[:feat_half1B.shape[0]] = feat_half1B
                    feat_half1B_aux[feat_half1B.shape[0]:] = feat_half1B[feat_half1B.shape[0]-1]
                    feat_half1B = feat_half1B_aux
                    print('Resized to: ' + str(feat_half1B.shape))
                    
                if feat_half2B.shape[0]*2 < feat_half2R.shape[0]:
                    print('Different shape')
                    print('Previous shape: ' + str(feat_half2B.shape))
                    feat_half2B_aux = np.zeros((feat_half2R.shape[0] // 2, feat_half2B.shape[1]))
                    feat_half2B_aux[:feat_half2B.shape[0]] = feat_half2B
                    feat_half2B_aux[feat_half2B.shape[0]:] = feat_half2B[feat_half2B.shape[0]-1]
                    feat_half2B = feat_half2B_aux
                    print('Resized to: ' + str(feat_half2B.shape))
                
                
                feat_half1B = feats2clip(torch.from_numpy(feat_half1B), stride=stride, clip_length=self.chunk_size) 
                feat_half1R = feats2clip(torch.from_numpy(feat_half1R), stride=stride * 2, clip_length=self.chunk_size * 2) 
                feat_half2B = feats2clip(torch.from_numpy(feat_half2B), stride=stride, clip_length=self.chunk_size) 
                feat_half2R = feats2clip(torch.from_numpy(feat_half2R), stride=stride * 2, clip_length=self.chunk_size * 2) 


            

            # Load labels
            labels = json.load(open(os.path.join(labels_path, game, self.labels)))
            if self.path != 'Baidu+ResNet':
                label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
                label_half1[:,0]=1 # those are BG classes
                label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
                label_half2[:,0]=1 # those are BG classes
            else:
                label_half1 = np.zeros((feat_half1B.shape[0], self.num_classes+1))
                label_half1[:,0]=1 # those are BG classes
                label_half2 = np.zeros((feat_half2B.shape[0], self.num_classes+1))
                label_half2[:,0]=1 # those are BG classes

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//stride>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//stride>=label_half2.shape[0]:
                    continue
                a = frame // stride
                if half == 1:
                    for i in range(self.chunk_size // stride):
                        label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                        label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1
                    #label_half1[max(a - self.chunk_size//stride + 1, 0) : (a + 1)][0] = 0 # not BG anymore

                if half == 2:
                    for i in range(self.chunk_size // stride):
                        label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                        label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1 # that's my class
            
            if self.path != 'Baidu+ResNet':
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
            else:
                self.game_feats1.append(feat_half1B)
                self.game_feats2.append(feat_half1R)
                self.game_feats1.append(feat_half2B)
                self.game_feats2.append(feat_half2R)
           
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
            
        if self.path != 'Baidu+ResNet':
            self.game_feats = np.concatenate(self.game_feats)
            self.game_labels = np.concatenate(self.game_labels)
            
            if augment == True:
                n_aug = 10000
                weights = np.array([0.01, 1/0.88, 1/0.76, 1/0.79, 1/0.70, 1/0.56, 1/0.58, 
                                    1/0.58, 1/0.71, 1/0.87, 1/0.85, 1/0.77, 1/0.62, 
                                    1/0.69, 1/0.89, 1/0.69, 1/0.08, 1/0.19])**2
                prob_ind = self.game_labels.dot(weights)
                
                i=0
                feat_aug_list = []
                y_aug_list = []
                while(i < n_aug):
                    i+=1
                    id1 = random.choices(np.arange(0, len(self.game_feats)), weights = prob_ind, k=1)
                    id2 = random.choices(np.arange(0, len(self.game_feats)), weights = prob_ind, k=1)
                    while(id1 == id2):
                        id2 = random.choices(np.arange(0, len(self.game_feats)), weights = prob_ind, k=1)
                    feat_aug, y_aug = mix_up(self.game_feats[id1], self.game_feats[id2], self.game_labels[id1], self.game_labels[id2])
                    feat_aug_list.append(feat_aug)
                    y_aug_list.append(y_aug)
                feat_aug_list = np.concatenate(feat_aug_list)
                y_aug_list = np.concatenate(y_aug_list)
                self.game_feats = np.concatenate((self.game_feats, feat_aug_list))
                self.game_labels = np.concatenate((self.game_labels, y_aug_list))
                
        else:
            self.game_feats1 = np.concatenate(self.game_feats1)
            self.game_feats2 = np.concatenate(self.game_feats2)
            self.game_labels = np.concatenate(self.game_labels)
        #self.game_labels = np.concatenate(self.game_labels)
        print(self.dict_event)
        class_weights1 = len(self.game_labels) / (self.game_labels.sum(axis = 0) * 2) 
        class_weights2 = len(self.game_labels) / ((len(self.game_labels) - self.game_labels.sum(axis = 0)) * 2)
        self.class_weights1 = class_weights1
        self.class_weights2 = class_weights2
        self.class_weights1[self.class_weights1 > 100] = 100
        print(self.class_weights1)
        print(self.class_weights2)
        #self.weights = (self.game_labels * class_weights).sum(axis = 1)
        #print(self.weights.shape)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        if self.path != 'Baidu+ResNet':
            return self.game_feats[index,:,:], self.game_labels[index,:]
        else:
            return self.game_feats1[index,:,:], self.game_feats2[index,:,:], self.game_labels[index,:]

    def __len__(self):
        if self.path != 'Baidu+ResNet':
            return len(self.game_feats)
        else:
            return len(self.game_feats1)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=1, 
                framerate=2, chunk_size=240):
        self.path = path
        
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.version = version
        self.split = split
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #for s in split:
        #    if s == "challenge":
        #        downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
        #    else:
        #        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
        baidu_path = '/data-net/datasets/SoccerNetv2/Baidu_features'
        baidu_name = 'baidu_soccer_embeddings.npy'
        resnet_path = '/home-net/axesparraguera/data/VGGFeatures'
        resnet_name = 'VGGish.npy'
        # Load features
        
        if self.path != 'Baidu+ResNet':
            feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1]) #for C3D non PCA
            feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1]) #for C3D non PCA
    
            # Load labels
            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))
        
        else:
            feat1_half1 = np.load(os.path.join(baidu_path, self.listGames[index], "1_" + baidu_name))
            feat1_half1 = feat1_half1.reshape(-1, feat1_half1.shape[-1])    #for C3D non PCA
            feat1_half2 = np.load(os.path.join(baidu_path, self.listGames[index], "2_" + baidu_name))
            feat1_half2 = feat1_half2.reshape(-1, feat1_half2.shape[-1])    #for C3D non PCA
            feat2_half1 = np.load(os.path.join(resnet_path, self.listGames[index], "1_" + resnet_name))
            feat2_half1 = feat2_half1.reshape(-1, feat2_half1.shape[-1])    #for C3D non PCA
            feat2_half2 = np.load(os.path.join(resnet_path, self.listGames[index], "2_" + resnet_name))
            feat2_half2 = feat2_half2.reshape(-1, feat2_half2.shape[-1])    #for C3D non PCA

            
            label_half1 = np.zeros((feat1_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat1_half2.shape[0], self.num_classes))
        
        # check if annoation exists
        if os.path.exists(os.path.join(labels_path, self.listGames[index], self.labels)):
        
            labels = json.load(open(os.path.join(labels_path, self.listGames[index], self.labels)))
            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if self.version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif self.version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1
                if self.path != 'Baidu+ResNet':
                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_half1[frame][label] = value
    
                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_half2[frame][label] = value
                else:
                    if half == 1:
                        frame = min(frame, feat1_half1.shape[0]-1)
                        label_half1[frame][label] = value
    
                    if half == 2:
                        frame = min(frame, feat1_half2.shape[0]-1)
                        label_half2[frame][label] = value

            
                
        if self.path != 'Baidu+ResNet':
            feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                            stride=1, off=int(self.chunk_size/2), 
                            clip_length=self.chunk_size)
    
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                            stride=1, off=int(self.chunk_size/2), 
                            clip_length=self.chunk_size)
            return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2
        
        else:
            
            if feat1_half1.shape[0]*2 > feat2_half1.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat2_half1.shape))
                feat2_half1_aux = np.zeros((feat1_half1.shape[0] * 2, feat2_half1.shape[1]))
                feat2_half1_aux[:feat2_half1.shape[0]] = feat2_half1
                feat2_half1_aux[feat2_half1.shape[0]:] = feat2_half1[feat2_half1.shape[0]-1]
                feat2_half1 = feat2_half1_aux
                print('Resized to: ' + str(feat2_half1.shape))
                
            if feat1_half2.shape[0]*2 > feat2_half2.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat2_half2.shape))
                feat2_half2_aux = np.zeros((feat1_half2.shape[0] * 2, feat2_half2.shape[1]))
                feat2_half2_aux[:feat2_half2.shape[0]] = feat2_half2
                feat2_half2_aux[feat2_half2.shape[0]:] = feat2_half2[feat2_half2.shape[0]-1]
                feat2_half2 = feat2_half2_aux
                print('Resized to: ' + str(feat2_half2.shape))
                
            if feat1_half1.shape[0]*2 < feat2_half1.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat1_half1.shape))
                feat1_half1_aux = np.zeros((feat2_half1.shape[0] // 2, feat1_half1.shape[1]))
                feat1_half1_aux[:feat1_half1.shape[0]] = feat1_half1
                feat1_half1_aux[feat1_half1.shape[0]:] = feat1_half1[feat1_half1.shape[0]-1]
                feat1_half1 = feat1_half1_aux
                print('Resized to: ' + str(feat1_half1.shape))
                
            if feat1_half2.shape[0]*2 < feat2_half2.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat1_half2.shape))
                feat1_half2_aux = np.zeros((feat2_half2.shape[0] // 2, feat1_half2.shape[1]))
                feat1_half2_aux[:feat1_half2.shape[0]] = feat1_half2
                feat1_half2_aux[feat1_half2.shape[0]:] = feat1_half2[feat1_half2.shape[0]-1]
                feat1_half2 = feat1_half2_aux
                print('Resized to: ' + str(feat1_half2.shape))
            
            feat1_half1 = feats2clip(torch.from_numpy(feat1_half1),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
            feat1_half2 = feats2clip(torch.from_numpy(feat1_half2),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
            feat2_half1 = feats2clip(torch.from_numpy(feat2_half1),
                                     stride=2, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size * 2)
            feat2_half2 = feats2clip(torch.from_numpy(feat2_half2),
                                     stride=2, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size * 2)
            
            if feat1_half1.shape[0] != feat2_half1.shape[0]:
                feat2_half1 = feat2_half1[:feat1_half1.shape[0]]
            if feat1_half2.shape[0] != feat2_half2.shape[0]:
                feat2_half2 = feat2_half2[:feat1_half2.shape[0]]
            
            return self.listGames[index], feat1_half1, feat2_half1, feat1_half2, feat2_half2, label_half1, label_half2

        
        

    def __len__(self):
        return len(self.listGames)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    dataset_Train = SoccerNetClips(path="/path/to/SoccerNet/" ,features="ResNET_PCA512.npy", split="train")
    print(len(dataset_Train))
    all_labels = []
    from tqdm import tqdm
    for i in tqdm(range(len(dataset_Train))):
        feats, labels = dataset_Train[i]
        all_labels.append(labels)
    all_labels = np.stack(all_labels)
    print(all_labels.shape)
    print(np.sum(all_labels,axis=0))
    print(np.sum(all_labels,axis=1))
        # print(feats.shape, labels)



    # train_loader = torch.utils.data.DataLoader(dataset_Train,
    #     batch_size=8, shuffle=True,
    #     num_workers=4, pin_memory=True)
    # for i, (feats, labels) in enumerate(train_loader):
    #     print(i, feats.shape, labels.shape)

    # dataset_Test = SoccerNetClipsTesting(path="/path/to/SoccerNet/" ,features="ResNET_PCA512.npy")
    # print(len(dataset_Test))
    # for i in range(2):
    #     feats1, feats2, labels1, labels2 = dataset_Test[i]
    #     print(feats1.shape)
    #     print(labels1.shape)
    #     print(feats2.shape)
    #     print(labels2.shape)
    #     # print(feats1[-1])
