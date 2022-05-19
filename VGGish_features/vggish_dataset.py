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
import moviepy.editor as mp
import soundfile as sf
import json
import random
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1
from mel_features import *
from vggish_input import *
from pydub import AudioSegment

print('starting program')

class GenerateWav(Dataset):
    def __init__(self, path, features="224p.mp3", split = ["train", "valid", "test", "challenge"]):
        self.path = path
        self.features = features
        self.listGames = getListGames(split)
        for game in tqdm(self.listGames):
            print(game)

            # Load wav audio file
            if not os.path.exists(os.path.join(self.path, game, "1_audio.wav")):
                try:
                    sound = AudioSegment.from_mp3(os.path.join(self.path, game, "1_" + self.features))
                    sound.export(os.path.join(self.path, game, "1_audio.wav"), format="wav")
                    
                    #my_clip_1 = mp.VideoFileClip(os.path.join(self.path, game, "1_" + self.features))
                    #my_clip_1.audio.write_audiofile(os.path.join(self.path, game, "1_audio.wav"))
                
                except:
                    print('Problem with following file:\n')
                    print(os.path.join(self.path, game, "1_" + self.features))
            
            
            if not os.path.exists(os.path.join(self.path, game, "2_audio.wav")):
                try:
                    sound = AudioSegment.from_mp3(os.path.join(self.path, game, "2_" + self.features))
                    sound.export(os.path.join(self.path, game, "2_audio.wav"), format="wav")
                    
                    #my_clip_2 = mp.VideoFileClip(os.path.join(self.path, game, "2_" + self.features))
                    #my_clip_2.audio.write_audiofile(os.path.join(self.path, game, "2_audio.wav"))
                
                except:
                    print('Problem with following file:\n')
                    print(os.path.join(self.path, game, "2_" + self.features))
        
        
        
class AudioFeatures(Dataset):
    def __init__(self, path, features="audio.mp3", split=["train", "valid", "test", "challenge"], version=2, 
                framerate=44100, chunk_size=42336, augment = False, only_labels = True):
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
        
        self.feats = list() 
        self.game_labels = list()

        # game_counter = 0
        stride = self.chunk_size #// 2
        
        
        for game in tqdm(self.listGames):
            #From mp3 to wav
            
            if only_labels:
                exist1 = os.path.exists(os.path.join(self.path, game, "1_audio.npy"))
                exist2 = os.path.exists(os.path.join(self.path, game, "2_audio.npy"))
                shape1 = 0
                shape2 = 0
                if exist1:
                    shape1 = np.load(os.path.join(self.path, game, "1_audio.npy")).shape[0]
                if exist2:
                    shape2 = np.load(os.path.join(self.path, game, "2_audio.npy")).shape[0]
                    
                if exist1 & exist2 & (shape1 > 100) & (shape2 > 100):
                    
                    print('Storing labels...')
                    if os.path.exists(os.path.join(labels_path, game, self.labels)):
                        labels = json.load(open(os.path.join(labels_path, game, self.labels)))
                        label_half1 = np.zeros((shape1, self.num_classes+1))
                        label_half1[:,0]=1 # those are BG classes
                        label_half2 = np.zeros((shape2, self.num_classes+1))
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
                                for i in range(-2, self.chunk_size // stride + 2):
                                    label_half1[min(max(a - self.chunk_size // stride + 1 + i, 0), shape1-1)][0] = 0 # not BG anymore
                                    label_half1[min(max(a - self.chunk_size // stride + 1 + i, 0), shape1-1)][label+1] = 1
                                    #label_half1[max(a - self.chunk_size//stride + 1, 0) : (a + 1)][0] = 0 # not BG anymore
            
                            if half == 2:
                                for i in range(-2, self.chunk_size // stride + 2):
                                    label_half2[min(max(a - self.chunk_size // stride + 1 + i, 0), shape2-1)][0] = 0 # not BG anymore
                                    label_half2[min(max(a - self.chunk_size // stride + 1 + i, 0), shape2-1)][label+1] = 1 # that's my class
                                    
                        #idx1 = (1 - (label_half1 == 0) * random.choices([0, 1], weights = [0.05, 0.95], k = len(label_half1))).astype('bool')
                        #idx2 = (1 - (label_half2 == 0) * random.choices([0, 1], weights = [0.05, 0.95], k = len(label_half2))).astype('bool')
            
                        #feat_half1 = feat_half1[idx1, :, :]
                        #label_half1 = label_half1[idx1]
                        #feat_half2 = feat_half2[idx2, :, :]
                        #label_half2 = label_half2[idx2]
                        
                        print(shape1)
                        print(label_half1.shape)
                        print(shape2)
                        print(label_half2.shape)
                        
                        
                        
                        
                        print('Storing npy files...')

                        np.save(os.path.join(self.path, game, "1_labels.npy"), label_half1)
                        np.save(os.path.join(self.path, game, "2_labels.npy"), label_half2)

                
                
            else:
                exist = os.path.exists(os.path.join(self.path, game, "1_audio.npy"))
                shape = 1
                if exist:
                    shape = np.load(os.path.join(self.path, game, "1_audio.npy")).shape[0]
                if not exist | (shape == 0):
                    self.features = "224p.mp3"
                    
                    if not os.path.exists(os.path.join(self.path, game, "1_audio.wav")):
                        try:
                            print('Converting from mp3 to wav')
                            sound = AudioSegment.from_mp3(os.path.join(self.path, game, "1_" + self.features))
                            sound.export(os.path.join(self.path, game, "1_audio.wav"), format="wav")
                            
                            #my_clip_1 = mp.VideoFileClip(os.path.join(self.path, game, "1_" + self.features))
                            #my_clip_1.audio.write_audiofile(os.path.join(self.path, game, "1_audio.wav"))
                        
                        except:
                            print('Problem with following file:\n')
                            print(os.path.join(self.path, game, "1_" + self.features))
                    
                    
                    if not os.path.exists(os.path.join(self.path, game, "2_audio.wav")):
                        try:
                            sound = AudioSegment.from_mp3(os.path.join(self.path, game, "2_" + self.features))
                            sound.export(os.path.join(self.path, game, "2_audio.wav"), format="wav")
                            
                            #my_clip_2 = mp.VideoFileClip(os.path.join(self.path, game, "2_" + self.features))
                            #my_clip_2.audio.write_audiofile(os.path.join(self.path, game, "2_audio.wav"))
                        
                        except:
                            print('Problem with following file:\n')
                            print(os.path.join(self.path, game, "2_" + self.features))
                    
                    
                    #From wav to npy           
                    
                    self.features = "audio.wav"
                    
                    if os.path.exists(os.path.join(self.path, game, "1_" + self.features)):
                        try:
                            # Load wav audio file
                            print('1st half wav to npy...')
                            feat_half1 = wavfile_to_examples(os.path.join(self.path, game, "1_" + self.features))
                            print('2nd half wav to npy...')
                            feat_half2 = wavfile_to_examples(os.path.join(self.path, game, "2_" + self.features))
                            
                            print('Features shape:')
                            print(feat_half1.shape)
                            print(feat_half2.shape)
                            
                            print('Deleting .wav file...')
                            if feat_half1.shape[0] > 0:
                                os.remove(os.path.join(self.path, game, "1_" + self.features))
                            else:
                                print('Empty features')
                            if feat_half2.shape[0] > 0:
                                os.remove(os.path.join(self.path, game, "2_" + self.features))
                            else:
                                print('Empty features')
                            
                            print('Saving npy features files')
                            np.save(os.path.join(self.path, game, "1_audio.npy"), feat_half1)
                            np.save(os.path.join(self.path, game, "2_audio.npy"), feat_half2)
                            
                            # Load labels
                            print('Lets focus on labels')
                            if os.path.exists(os.path.join(labels_path, game, self.labels)):
                                labels = json.load(open(os.path.join(labels_path, game, self.labels)))
                                label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
                                label_half1[:,0]=1 # those are BG classes
                                label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
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
                                            
                                #idx1 = (1 - (label_half1 == 0) * random.choices([0, 1], weights = [0.05, 0.95], k = len(label_half1))).astype('bool')
                                #idx2 = (1 - (label_half2 == 0) * random.choices([0, 1], weights = [0.05, 0.95], k = len(label_half2))).astype('bool')
                    
                                #feat_half1 = feat_half1[idx1, :, :]
                                #label_half1 = label_half1[idx1]
                                #feat_half2 = feat_half2[idx2, :, :]
                                #label_half2 = label_half2[idx2]
                                
                                
                                print(label_half1.shape)
                                
                                print(label_half2.shape)
                                
                                
                                
                                
                                print('Storing npy files...')
        
                                np.save(os.path.join(self.path, game, "1_labels.npy"), label_half1)
                                np.save(os.path.join(self.path, game, "2_labels.npy"), label_half2)
                            
                            
                            
                            '''
                            self.feats.append(feat_half1)
                            self.feats.append(feat_half2)
                            self.game_labels.append(label_half1)
                            self.game_labels.append(label_half2)
                            '''
                            
                        except:
                            print('Not correct wav file')
                        
                    else:
                        print('Match without audio features')
                else:
                    print('Already has audio features')
            
        #self.feats = np.concatenate(self.feats)
        #self.game_labels = np.concatenate(self.game_labels)
        
        #print(self.feats.shape)
        #print(self.game_labels.shape)
        
#GenerateWav('/data-net/datasets/SoccerNetv2/videos_lowres')

audios = AudioFeatures('/data-local/data1-hdd/axesparraguera/vggish')
