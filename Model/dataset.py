'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file generates samples to feed the HMTAS model
'''

from torch.utils.data import Dataset
import numpy as np
import random
import os
import time
import pickle
import blosc
from tqdm import tqdm
import torch
import logging
import json
import random
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

'Function for mix-up data augmentation'
def mix_up(feat1, feat2, y1, y2):
    lam = np.random.beta(a = 0.2, b=0.2, size=1)
    feat_new = feat1 * lam + feat2 * (1-lam)
    y_new = y1 * lam + y2 * (1-lam)
    feat_new_list = [l.tolist() for l in feat_new]
    y_new_list = (y_new).tolist()
    return feat_new_list, y_new_list


'Function to generate features from clips'
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


'Class to generate samples and store them in RAM to train the HMTAS model'
class SoccerNetClips(Dataset):
    def __init__(self, path_baidu = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                 path_audio = '/data-local/data1-hdd/axesparraguera/vggish', 
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2", 
                 features_baidu = 'baidu_soccer_embeddings.npy',
                 features_audio = 'featA2.npy', split=["train"], version=1, 
                framerate=2, chunk_size=240, augment = False):

        labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
        self.listGames = getListGames(split)
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
        framerate2 = 1
        stride = self.chunk_size #// 2
        for game in tqdm(self.listGames):
            
            feat_half1B = np.load(os.path.join(path_baidu, game, "1_" + features_baidu))
            feat_half1B = feat_half1B.reshape(-1, feat_half1B.shape[-1])
            feat_half1R = np.load(os.path.join(path_audio, game, "1_" + features_audio))
            feat_half1R = feat_half1R.reshape(-1, feat_half1R.shape[-1])
            feat_half2B = np.load(os.path.join(path_baidu, game, "2_" + features_baidu))
            feat_half2B = feat_half2B.reshape(-1, feat_half2B.shape[-1])
            feat_half2R = np.load(os.path.join(path_audio, game, "2_" + features_audio))
            feat_half2R = feat_half2R.reshape(-1, feat_half2R.shape[-1])
                
            if feat_half1B.shape[0]*framerate2 > feat_half1R.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat_half1R.shape))
                feat_half1R_aux = np.zeros((feat_half1B.shape[0] * framerate2, feat_half1R.shape[1]))
                feat_half1R_aux[:feat_half1R.shape[0]] = feat_half1R
                feat_half1R_aux[feat_half1R.shape[0]:] = feat_half1R[feat_half1R.shape[0]-1]
                feat_half1R = feat_half1R_aux
                print('Resized to: ' + str(feat_half1R.shape))
                    
            if feat_half2B.shape[0]*framerate2 > feat_half2R.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat_half2R.shape))
                feat_half2R_aux = np.zeros((feat_half2B.shape[0] * framerate2, feat_half2R.shape[1]))
                feat_half2R_aux[:feat_half2R.shape[0]] = feat_half2R
                feat_half2R_aux[feat_half2R.shape[0]:] = feat_half2R[feat_half2R.shape[0]-1]
                feat_half2R = feat_half2R_aux
                print('Resized to: ' + str(feat_half2R.shape))
                    
            if feat_half1B.shape[0]*framerate2 < feat_half1R.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat_half1B.shape))
                feat_half1B_aux = np.zeros((feat_half1R.shape[0] // framerate2, feat_half1B.shape[1]))
                feat_half1B_aux[:feat_half1B.shape[0]] = feat_half1B
                feat_half1B_aux[feat_half1B.shape[0]:] = feat_half1B[feat_half1B.shape[0]-1]
                feat_half1B = feat_half1B_aux
                print('Resized to: ' + str(feat_half1B.shape))
                    
            if feat_half2B.shape[0]*framerate2 < feat_half2R.shape[0]:
                print('Different shape')
                print('Previous shape: ' + str(feat_half2B.shape))
                feat_half2B_aux = np.zeros((feat_half2R.shape[0] // framerate2, feat_half2B.shape[1]))
                feat_half2B_aux[:feat_half2B.shape[0]] = feat_half2B
                feat_half2B_aux[feat_half2B.shape[0]:] = feat_half2B[feat_half2B.shape[0]-1]
                feat_half2B = feat_half2B_aux
                print('Resized to: ' + str(feat_half2B.shape))
                
                
            feat_half1B = feats2clip(torch.from_numpy(feat_half1B), stride=stride, clip_length=self.chunk_size) 
            feat_half1R = feats2clip(torch.from_numpy(feat_half1R), stride=stride * framerate2, clip_length=self.chunk_size * framerate2) 
            feat_half2B = feats2clip(torch.from_numpy(feat_half2B), stride=stride, clip_length=self.chunk_size) 
            feat_half2R = feats2clip(torch.from_numpy(feat_half2R), stride=stride * framerate2, clip_length=self.chunk_size * framerate2) 


            

            # Load labels
            labels = json.load(open(os.path.join(path_labels, game, self.labels)))


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
            


            self.game_feats1.append(feat_half1B)
            self.game_feats2.append(feat_half1R)
            self.game_feats1.append(feat_half2B)
            self.game_feats2.append(feat_half2R)
           
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
                        

        self.game_feats1 = np.concatenate(self.game_feats1)
        self.game_feats2 = np.concatenate(self.game_feats2)
        self.game_labels = np.concatenate(self.game_labels)
        #self.game_labels = np.concatenate(self.game_labels)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats1[index,:,:], self.game_feats2[index,:,:], self.game_labels[index,:]

    def __len__(self):

        return len(self.game_feats1)
        
        
        
'Class to generate samples and store them to train the HMTAS model'
class SoccerNetClipsTrain(Dataset):
    def __init__(self, path_baidu = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                 path_audio = '/data-local/data1-hdd/axesparraguera/vggish', 
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2", 
                 path_store = "/data-local/data3-ssd/axesparraguera",
                 features_baidu = 'baidu_soccer_embeddings.npy',
                 features_audio = 'featA2.npy', stride = 2, split=["train"], version=2, 
                framerate=1, chunk_size=7, augment = False, store = True):
        
        self.path_baidu = path_baidu
        self.path_audio = path_audio
        self.path_labels = path_labels
        self.path_store = path_store
        self.features_baidu = features_baidu
        self.features_audio = features_audio
        
        self.listGames = getListGames(split)
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
        
        self.stride = stride
        framerate2 = 1
        
        if store:
        
            self.path_list = []
            self.n_samples = []

            for game in tqdm(self.listGames):
                # Load features
                correct_audio = os.path.exists(os.path.join(self.path_audio, game, "1_" + self.features_audio))
                if correct_audio:
                    #audio = np.load(os.path.join(self.path_audio, game, "1_audio.npy"))
                    n_audio = 1000 #audio.shape[0]
                    if n_audio > 100:
                
                        feat_half1B = np.load(os.path.join(self.path_baidu, game, "1_" + self.features_baidu))
                        feat_half1B = feat_half1B.reshape(-1, feat_half1B.shape[-1])
                        feat_half1A = np.load(os.path.join(self.path_audio, game, "1_" + self.features_audio))
                        feat_half1A = feat_half1A.reshape(-1, feat_half1A.shape[-1])
                        feat_half2B = np.load(os.path.join(self.path_baidu, game, "2_" + self.features_baidu))
                        feat_half2B = feat_half2B.reshape(-1, feat_half2B.shape[-1])
                        feat_half2A = np.load(os.path.join(self.path_audio, game, "2_" + self.features_audio))
                        feat_half2A = feat_half2A.reshape(-1, feat_half2A.shape[-1])
                            
                        if feat_half1B.shape[0]*framerate2 > feat_half1A.shape[0]:
                            print('Different shape')
                            print('Previous shape: ' + str(feat_half1A.shape))
                            feat_half1A_aux = np.zeros((feat_half1B.shape[0] * framerate2, feat_half1A.shape[1]))
                            feat_half1A_aux[:feat_half1A.shape[0]] = feat_half1A
                            feat_half1A_aux[feat_half1A.shape[0]:] = feat_half1A[feat_half1A.shape[0]-1]
                            feat_half1A = feat_half1A_aux
                            print('Resized to: ' + str(feat_half1A.shape))
                                
                        if feat_half2B.shape[0]*framerate2 > feat_half2A.shape[0]:
                            print('Different shape')
                            print('Previous shape: ' + str(feat_half2A.shape))
                            feat_half2A_aux = np.zeros((feat_half2B.shape[0] * framerate2, feat_half2A.shape[1]))
                            feat_half2A_aux[:feat_half2A.shape[0]] = feat_half2A
                            feat_half2A_aux[feat_half2A.shape[0]:] = feat_half2A[feat_half2A.shape[0]-1]
                            feat_half2A = feat_half2A_aux
                            print('Resized to: ' + str(feat_half2A.shape))
                                
                        if feat_half1B.shape[0]*framerate2 < feat_half1A.shape[0]:
                            print('Different shape')
                            print('Previous shape: ' + str(feat_half1B.shape))
                            feat_half1B_aux = np.zeros((feat_half1A.shape[0] // framerate2, feat_half1B.shape[1]))
                            feat_half1B_aux[:feat_half1B.shape[0]] = feat_half1B
                            feat_half1B_aux[feat_half1B.shape[0]:] = feat_half1B[feat_half1B.shape[0]-1]
                            feat_half1B = feat_half1B_aux
                            print('Resized to: ' + str(feat_half1B.shape))
                                
                        if feat_half2B.shape[0]*framerate2 < feat_half2A.shape[0]:
                            print('Different shape')
                            print('Previous shape: ' + str(feat_half2B.shape))
                            feat_half2B_aux = np.zeros((feat_half2A.shape[0] // framerate2, feat_half2B.shape[1]))
                            feat_half2B_aux[:feat_half2B.shape[0]] = feat_half2B
                            feat_half2B_aux[feat_half2B.shape[0]:] = feat_half2B[feat_half2B.shape[0]-1]
                            feat_half2B = feat_half2B_aux
                            print('Resized to: ' + str(feat_half2B.shape))
                            
                        feat_half1B = feats2clip(torch.from_numpy(feat_half1B), stride=stride, clip_length=self.chunk_size) 
                        feat_half1A = feats2clip(torch.from_numpy(feat_half1A), stride=stride * framerate2, clip_length=self.chunk_size * framerate2) 
                        feat_half2B = feats2clip(torch.from_numpy(feat_half2B), stride=stride, clip_length=self.chunk_size) 
                        feat_half2A = feats2clip(torch.from_numpy(feat_half2A), stride=stride * framerate2, clip_length=self.chunk_size * framerate2) 
                                   
            
                        # Load labels
                        labels = json.load(open(os.path.join(self.path_labels, game, self.labels)))
            
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
                        
                        path = os.path.join(self.path_store, game)
                        exists = os.path.exists(path)
                        if not exists:
                            os.makedirs(path)
                        
                        #Half1
                        feat_half1B = feat_half1B.numpy()
                        feat_half1A = feat_half1A.numpy()
                        print('Storing 1st half chunks...')
                        for i in range(feat_half1B.shape[0]):
                            np.save(path + '/half1_chunk' + str(i) + '_featuresB.npy', feat_half1B[i, :, :])
                            np.save(path + '/half1_chunk' + str(i) + '_featuresA.npy', feat_half1A[i, :, :])
                            np.save(path + '/half1_chunk' + str(i) + '_labels.npy', label_half1[i, :])
                            self.path_list.append(path + '/half1_chunk' + str(i) + '_')
                        '''
                        with open(path + '/half1_chunk' + '_featuresB.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(feat_half1B)))    
                        with open(path + '/half1_chunk' + '_featuresA.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(feat_half1A)))             
                        with open(path + '/half1_chunk' + '_labels.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(label_half1)))
                        
                            
                        self.path_list.append(path + '/half1_chunk' + '_')
                        self.n_samples.append(feat_half1B.shape[0])
                        
                        '''
                            
                        #Half2
                        print('Storing 2nd half chunks...')
                        
                        feat_half2B = feat_half2B.numpy()
                        feat_half2A = feat_half2A.numpy()
                        for i in range(feat_half2B.shape[0]):
                            np.save(path + '/half2_chunk' + str(i) + '_featuresB.npy', feat_half2B[i, :, :])
                            np.save(path + '/half2_chunk' + str(i) + '_featuresA.npy', feat_half2A[i, :, :])
                            np.save(path + '/half2_chunk' + str(i) + '_labels.npy', label_half2[i, :])
                            self.path_list.append(path + '/half2_chunk' + str(i) + '_')
                        '''
                        with open(path + '/half2_chunk' + '_featuresB.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(feat_half2B)))    
                        with open(path + '/half2_chunk' + '_featuresA.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(feat_half2A)))             
                        with open(path + '/half2_chunk' + '_labels.dat', 'wb') as f:
                            f.write(blosc.compress(pickle.dumps(label_half2)))  
                                       
                        self.path_list.append(path + '/half2_chunk' + '_')
                        self.n_samples.append(feat_half2B.shape[0])
                        
                        '''
                    else:
                        print('Not correct game: ' + game)
                else:
                    print('Not correct game: ' + game)
                    
            with open(self.path_store + '/chunk_list.pkl', 'wb') as f:
                pickle.dump(self.path_list, f)  
            
            #with open(self.path_store + '/n_samples.pkl', 'wb') as f:
            #    pickle.dump(self.n_samples, f)
            #self.weights = (self.game_labels * class_weights).sum(axis = 1)
            #print(self.weights.shape)
        
        else:
            with open(self.path_store + '/chunk_list.pkl', 'rb') as f:
                self.path_list = pickle.load(f)
            #with open(self.path_store + '/n_samples.pkl', 'rb') as f:
            #    self.n_samples = pickle.load(f)
        '''       
        self.idx2path = dict()
        cumsum = np.array(np.cumsum(self.n_samples))
        for i in range(np.array(self.n_samples).sum()):
            #Index of the match
            j = (i >= cumsum).sum()
            path = self.path_list[j]
            idx2 = i - (j > 0).astype(int) * cumsum[j-1] #index inside match
            self.idx2path[i] = [path, idx2]          
        '''


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        #featB = np.load(self.path_list[index] + 'featuresB.npy')
        #featA = np.load(self.path_list[index] + 'featuresA.npy')
        #labels = np.load(self.path_list[index] + 'labels.npy')
        '''
        path, idx = self.idx2path[index]
        with open(path + 'featuresB.dat', "rb") as f:
            featB = pickle.loads(blosc.decompress(f.read())) 
        f.close()
        with open(path + 'featuresA.dat', "rb") as f:
            featA = pickle.loads(blosc.decompress(f.read()))
        f.close()
        with open(path + 'labels.dat', "rb") as f:
            labels = pickle.loads(blosc.decompress(f.read()))
        f.close()
        
        featBret = featB[idx, :, :]
        featAret = featA[idx, :, :]
        labelsret = labels[idx, :]
        '''
        path = self.path_list[index]
        return torch.from_numpy(np.load(path + 'featuresB.npy')), torch.from_numpy(np.load(path + 'featuresA.npy')), np.load(path + 'labels.npy')
        '''
        else:
            #Create dictionary with all the indexes for each path
            path_id = dict()
            for ind in index:
                path, idx = self.idx2path[ind]
                if path not in path_id.keys():
                    path_id[path] = [idx]
                else:
                    path_id[path].append(idx)
                    
            #For each different path read all different samples            
            
            i = 0
            for path in path_id.keys():                 
                with open(path + 'featuresB.dat', "rb") as f:
                    featBidx = pickle.loads(blosc.decompress(f.read()))[path_id[path], :, :]  
                with open(path + 'featuresA.dat', "rb") as f:
                    featAidx = pickle.loads(blosc.decompress(f.read()))[path_id[path], :, :] 
                with open(path + 'labels.dat', "rb") as f:
                    labelsidx = pickle.loads(blosc.decompress(f.read()))[path_id[path], :] 
                    
                if i == 0:
                    featB = featBidx
                    featA = featAidx
                    labels = labelsidx
                    i += 1     

                else:
                    featB = torch.cat((featB, featBidx))
                    featA = torch.cat((featA, featAidx))
                    labels = np.concatenate((labels, labelsidx))


        
        return featB, featA, labels
        '''
    def __len__(self):
        
        return(len(self.path_list))
        #return len(self.path_list)


'Class to generate the samples for the test part'
class SoccerNetClipsTesting(Dataset):
    def __init__(self, path_baidu = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                 path_audio = '/data-local/data1-hdd/axesparraguera/vggish', 
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2", 
                 path_store = "/data-local/data3-ssd/axesparraguera",
                 features_baidu = 'baidu_soccer_embeddings.npy',
                 features_audio = 'featA2.npy', split=["test"], version=1, 
                framerate=2, chunk_size=240):
        
        self.path_baidu = path_baidu
        self.path_labels = path_labels
        self.path_audio = path_audio
        self.features_baidu = features_baidu
        self.features_audio = features_audio
        self.listGames = getListGames(split)
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
        
        

        feat1_half1 = np.load(os.path.join(self.path_baidu, self.listGames[index], "1_" + self.features_baidu))
        feat1_half1 = feat1_half1.reshape(-1, feat1_half1.shape[-1])    #for C3D non PCA
        feat1_half2 = np.load(os.path.join(self.path_baidu, self.listGames[index], "2_" + self.features_baidu))
        feat1_half2 = feat1_half2.reshape(-1, feat1_half2.shape[-1])    #for C3D non PCA
        feat2_half1 = np.load(os.path.join(self.path_audio, self.listGames[index], "1_" + self.features_audio))
        feat2_half1 = feat2_half1.reshape(-1, feat2_half1.shape[-1])    #for C3D non PCA
        feat2_half2 = np.load(os.path.join(self.path_audio, self.listGames[index], "2_" + self.features_audio))
        feat2_half2 = feat2_half2.reshape(-1, feat2_half2.shape[-1])    #for C3D non PCA

            
        label_half1 = np.zeros((feat1_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat1_half2.shape[0], self.num_classes))
        
        # check if annoation exists
        if os.path.exists(os.path.join(self.path_labels, self.listGames[index], self.labels)):
        
            labels = json.load(open(os.path.join(self.path_labels, self.listGames[index], self.labels)))
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

                if half == 1:
                    frame = min(frame, feat1_half1.shape[0]-1)
                    label_half1[frame][label] = value
    
                if half == 2:
                    frame = min(frame, feat1_half2.shape[0]-1)
                    label_half2[frame][label] = value
        
            
        if feat1_half1.shape[0]*self.framerate > feat2_half1.shape[0]:
            feat2_half1_aux = np.zeros((feat1_half1.shape[0] * self.framerate, feat2_half1.shape[1]))
            feat2_half1_aux[:feat2_half1.shape[0]] = feat2_half1
            feat2_half1_aux[feat2_half1.shape[0]:] = feat2_half1[feat2_half1.shape[0]-1]
            feat2_half1 = feat2_half1_aux
                
        if feat1_half2.shape[0]*self.framerate > feat2_half2.shape[0]:
            feat2_half2_aux = np.zeros((feat1_half2.shape[0] * self.framerate, feat2_half2.shape[1]))
            feat2_half2_aux[:feat2_half2.shape[0]] = feat2_half2
            feat2_half2_aux[feat2_half2.shape[0]:] = feat2_half2[feat2_half2.shape[0]-1]
            feat2_half2 = feat2_half2_aux
                
        if feat1_half1.shape[0]*self.framerate < feat2_half1.shape[0]:
            feat1_half1_aux = np.zeros((feat2_half1.shape[0] // self.framerate, feat1_half1.shape[1]))
            feat1_half1_aux[:feat1_half1.shape[0]] = feat1_half1
            feat1_half1_aux[feat1_half1.shape[0]:] = feat1_half1[feat1_half1.shape[0]-1]
            feat1_half1 = feat1_half1_aux
                
        if feat1_half2.shape[0]*self.framerate < feat2_half2.shape[0]:
            feat1_half2_aux = np.zeros((feat2_half2.shape[0] // self.framerate, feat1_half2.shape[1]))
            feat1_half2_aux[:feat1_half2.shape[0]] = feat1_half2
            feat1_half2_aux[feat1_half2.shape[0]:] = feat1_half2[feat1_half2.shape[0]-1]
            feat1_half2 = feat1_half2_aux
            
        feat1_half1 = feats2clip(torch.from_numpy(feat1_half1),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
        feat1_half2 = feats2clip(torch.from_numpy(feat1_half2),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
        feat2_half1 = feats2clip(torch.from_numpy(feat2_half1),
                                     stride=self.framerate, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size * self.framerate)
        feat2_half2 = feats2clip(torch.from_numpy(feat2_half2),
                                     stride=self.framerate, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size * self.framerate)
            
        if feat1_half1.shape[0] != feat2_half1.shape[0]:
            feat2_half1 = feat2_half1[:feat1_half1.shape[0]]
        if feat1_half2.shape[0] != feat2_half2.shape[0]:
            feat2_half2 = feat2_half2[:feat1_half2.shape[0]]
            
        return self.listGames[index], feat1_half1, feat2_half1, feat1_half2, feat2_half2, label_half1, label_half2

        
        

    def __len__(self):
        return len(self.listGames)


'Main code of dataset'
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


    
class TrainEnsemble(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n = len(self.features)
        
        
    def __getitem__(self, index):
        return self.features[index, :, :], self.labels[index, :]
    
    def __len__(self):
        return self.n
