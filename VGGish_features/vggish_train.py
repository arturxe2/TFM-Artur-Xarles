'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file trains the VGGish model using the previously created samples
'''

from __future__ import print_function
from SoccerNet.Evaluation.utils import AverageMeter
import time
from random import shuffle
import numpy as np
from vggish_torch import *
import torch.nn as nn
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm
import torch
import logging
import json
from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from loss import NLLLoss_audio



'Class for the train samples'
class TrainVGGish(Dataset):
    def __init__(self, path="/data-local/data1-hdd/axesparraguera/vggish", features="audio.npy", labels="labels.npy", 
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
            

            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            labels_half1 = np.load(os.path.join(self.path, game, "1_" + self.labels))
            labels_half2 = np.load(os.path.join(self.path, game, "2_" + self.labels))
                
            idx1 = np.arange(0, labels_half1.shape[0])[(1 - (labels_half1[:, 0] == 1) * random.choices([0, 1], weights = [0.9, 0.1], k = len(labels_half1))).astype('bool')]
            idx2 = np.arange(0, labels_half2.shape[0])[(1 - (labels_half2[:, 0] == 1) * random.choices([0, 1], weights = [0.9, 0.1], k = len(labels_half2))).astype('bool')]
            
            if labels_half1.shape[0] == 0:
                print('Game without examples: ' + game)
               
            self.game_feats.append(feat_half1[idx1, :, :])
            self.game_feats.append(feat_half2[idx2, :, :])
            self.game_labels.append(labels_half1[idx1, :])
            self.game_labels.append(labels_half2[idx2, :])
                

                
        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)
        self.n = self.game_feats.shape[0]
        
        
    def __getitem__(self, index):
        return self.game_feats[index, :, :], self.game_labels[index, :]
    
    def __len__(self):
        return self.n


'Trainer of the model'        
def trainer(path, train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            #scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")
    training_stage = 0

    best_loss = 9000000

    n_bad_epochs = 0
    for epoch in range(max_epochs):
        if n_bad_epochs >= patience:
            break

        
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        # train for one epoch
        loss_training = train(path, train_loader, model, criterion, 
                              optimizer, epoch + 1, training_stage = training_stage,
                              train=True)

        # evaluate on validation set
        loss_validation = train(
            path, val_loader, model, criterion, optimizer, epoch + 1, 
            training_stage = training_stage, train=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = (loss_validation < best_loss)
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        torch.save(state, best_model_path)
        if is_better:
            n_bad_epochs = 0
            print('Saving new model')
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1
        # Test the model on the validation set


        # Reduce LR on Plateau after patience reached
        #prevLR = optimizer.param_groups[0]['lr']
        #scheduler.step(loss_validation)
        #currLR = optimizer.param_groups[0]['lr']
        #if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
        #    logging.info("Plateau Reached!")

        #if (prevLR < 2 * scheduler.eps and
        #        scheduler.num_bad_epochs >= scheduler.patience):
        #    logging.info(
        #        "Plateau Reached and no more reduction -> Exiting Loop")
        #    break

    return     

'Train for the VGGish model'
def train(path,
          dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          training_stage = 0,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:

        model.train()

    else:
        model.eval()

    end = time.time()
    
    #Potser al fer cuda() hi ha el problema
    
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.cuda()
            labels = labels.cuda()
            # compute output
            output = model(feats)
    
            # hand written NLL criterion
            loss = criterion(labels, output)
    
            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))
    
            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
                optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)
        
    return loss.avg


'Main part of the code'
if __name__ == '__main__':
    'URLs to download weights and pca parameters'
    model_urls = {
        'vggish': 'https://github.com/harritaylor/torchvggish/'
                  'releases/download/v0.1/vggish-10086976.pth',
        'pca': 'https://github.com/harritaylor/torchvggish/'
               'releases/download/v0.1/vggish_pca_params-970ea276.pth'
    }

    'Get the samples with the class TrainVGGish'    
    dataset_Train = TrainVGGish()
    dataset_val = TrainVGGish(split=["test"])
    'Generate loaders'
    train_loader = torch.utils.data.DataLoader(dataset_Train,
        batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False,
                                             num_workers=1, pin_memory=True)

    'Define model, optimizer and criterion'
    model = VGGish(urls = model_urls, pretrained = True, preprocess = False, postprocess=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, 
                                betas=(0.9, 0.999), eps=1e-08, 
                                weight_decay=1e-5, amsgrad=True)
    criterion = NLLLoss_audio()
    'Train model'
    trainer('', train_loader, val_loader, val_loader, 
            model, optimizer, criterion, patience=5,
            model_name='final_model',
            max_epochs=8, evaluation_frequency=2)
    
    
print('Finished Training')
    