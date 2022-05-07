
r"""A simple demonstration of running VGGish in training mode.
This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.
For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.
Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100
  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function
from SoccerNet.Evaluation.utils import AverageMeter
import time

from random import shuffle

import numpy as np


#import vggish_input
#import vggish_params
#import vggish_slim
from vggish_torch import *
import torch.nn as nn

from torch.utils.data import Dataset

import random
# import pandas as pd
import os



from tqdm import tqdm
# import utils

import torch

import logging
import json
from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from loss import NLLLoss_weights




class TrainVGGish(Dataset):
    def __init__(self, path="/data-net/datasets/SoccerNetv2/videos_lowres", features="audio.npy", labels="labels.npy", 
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
            if i < 10:

                # Load features
                feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
                feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
                labels_half1 = np.load(os.path.join(self.path, game, "1_" + self.labels))
                labels_half2 = np.load(os.path.join(self.path, game, "2_" + self.labels))
        
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(labels_half1)
                self.game_labels.append(labels_half2)
                
                #except:
                    #print('Not npy file')
                
        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)

        self.n = self.game_feats.shape[0]
        
        
    def __getitem__(self, index):
        return self.game_feats[index, :, :], self.game_lables[index, :]
    
    def __len__(self):
        return self.n

        
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

    best_loss = 9e99

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
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1
        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                path,
                val_metric_loader,
                model,
                model_name)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))

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
        
    return loss



if __name__ == '__main__':

    model = get_vggish(with_classifier=True, pretrained=True)
    model.classifier._modules['2'] = nn.Linear(100, 18)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, 
                                betas=(0.9, 0.999), eps=1e-08, 
                                weight_decay=1e-5, amsgrad=True)
    criterion = NLLLoss_weights()
    dataset_Train = TrainVGGish()
    dataset_val = TrainVGGish(split=["test"])
    train_loader = torch.utils.data.DataLoader(dataset_Train,
        batch_size=128, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=True)
    print(model)
    
    trainer('', train_loader, val_loader, val_loader, 
            model, optimizer, criterion, patience=5,
            model_name='model',
            max_epochs=5, evaluation_frequency=2)

print('Finished Training')
    