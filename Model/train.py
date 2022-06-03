import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np
import math
import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1
from model import Model, EnsembleModel
from dataset import SoccerNetClipsTesting, feats2clip, TrainEnsemble
from loss import NLLLoss_weights_ensemble
import random




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
            if training_stage == 0:
                training_stage += 1
                break
            else:
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
        
        if path != 'Baidu+ResNet':
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
        else:

            for i, (feats1, feats2, labels) in t:
                
                # measure data loading time
                data_time.update(time.time() - end)
                feats1 = feats1.cuda()
                feats2 = feats2.cuda()
                labels = labels.cuda()
                # compute output
                outputs_mix, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feats1, feats2)
        
                # hand written NLL criterion
                if training_stage == 0:
                    lossF = criterion(labels, outputs_mix)
                    lossA = criterion(labels, outputsA)
                    lossB1 = criterion(labels, outputsB1)
                    lossB2 = criterion(labels, outputsB2)
                    lossB3 = criterion(labels, outputsB3)
                    lossB4 = criterion(labels, outputsB4)
                    lossB5 = criterion(labels, outputsB5)
                    
                    loss = lossF #+ 0.05 * lossA + 0.05 * lossB1 + 0.05 * lossB2 + 0.05 * lossB3 + 0.05 * lossB4 + 0.05 * lossB5
                else:
                    loss = criterion(labels, outputs_mix)
                
                # measure accuracy and record loss
                losses.update(loss.item(), feats1.size(0) + feats2.size(0))
        
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
    if (training_stage == 0) & (path == 'Baidu+ResNet'):
        print('Total loss: ' + str(lossF))
        print('Audio loss: ' + str(lossA))
        print('Baidu1 loss: ' + str(lossB1))
        print('Baidu2 loss: ' + str(lossB2))
        print('Baidu3 loss: ' + str(lossB3))
        print('Baidu4 loss: ' + str(lossB4))
        print('Baidu5 loss: ' + str(lossB5))
    return losses.avg


def test(path, dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        if path != 'Baidu+ResNet':
            for i, (feats, labels) in t:
                # measure data loading time
                data_time.update(time.time() - end)
                feats = feats.cuda()
                # labels = labels.cuda()
    
                # print(feats.shape)
                # feats=feats.unsqueeze(0)
                # print(feats.shape)
    
                # compute output
                output = model(feats)
    
                all_labels.append(labels.detach().numpy())
                all_outputs.append(output.cpu().detach().numpy())
    
                batch_time.update(time.time() - end)
                end = time.time()
    
                desc = f'Test (cls): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)
        else:
            for i, (feats1, feats2, labels) in t:
                # measure data loading time
                data_time.update(time.time() - end)
                feats1 = feats1.cuda()
                feats2 = feats2.cuda()
                # labels = labels.cuda()
    
                # print(feats.shape)
                # feats=feats.unsqueeze(0)
                # print(feats.shape)
    
                # compute output
                output, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feats1, feats2)
    
                all_labels.append(labels.detach().numpy())
                all_outputs.append(output.cpu().detach().numpy())
    
                batch_time.update(time.time() - end)
                end = time.time()
    
                desc = f'Test (cls): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    # t.set_description()
    # print(AP)
    mAP = np.mean(AP)
    print(mAP, AP)
    # a_mAP = average_mAP(spotting_grountruth, spotting_predictions, model.framerate)
    # print("Average-mAP: ", a_mAP)

    return mAP

def testSpotting(path, dataloader, model, model_name, overwrite=True, NMS_window=30, NMS_threshold=0.5):

    split = '_'.join(dataloader.dataset.split)
    # print(split)
    output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    output_folder = f"outputs_{split}"

    if not os.path.exists(output_results) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()

        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        model.eval()

        count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

        end = time.time()
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            #Spotting if not Baidu+ResNet
            if path != 'Baidu+ResNet':
                for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
                    data_time.update(time.time() - end)
    
                    # Batch size of 1
                    game_ID = game_ID[0]
                    feat_half1 = feat_half1.squeeze(0)
                    label_half1 = label_half1.float().squeeze(0)
                    feat_half2 = feat_half2.squeeze(0)
                    label_half2 = label_half2.float().squeeze(0)
    
                    # Compute the output for batches of frames
                    BS = 256
                    timestamp_long_half_1 = []
                    for b in range(int(np.ceil(len(feat_half1)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat_half1) else len(feat_half1)-1
                        feat = feat_half1[start_frame:end_frame].cuda()
                        output = model(feat).cpu().detach().numpy()
                        timestamp_long_half_1.append(output)
                    timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)
    
                    timestamp_long_half_2 = []
                    for b in range(int(np.ceil(len(feat_half2)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat_half2) else len(feat_half2)-1
                        feat = feat_half2[start_frame:end_frame].cuda()
                        output = model(feat).cpu().detach().numpy()
                        timestamp_long_half_2.append(output)
                    timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)
    
    
                    timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                    timestamp_long_half_2 = timestamp_long_half_2[:, 1:]
    
                    spotting_grountruth.append(torch.abs(label_half1))
                    spotting_grountruth.append(torch.abs(label_half2))
                    spotting_grountruth_visibility.append(label_half1)
                    spotting_grountruth_visibility.append(label_half2)
                    spotting_predictions.append(timestamp_long_half_1)
                    spotting_predictions.append(timestamp_long_half_2)
                    # segmentation_predictions.append(segmentation_long_half_1)
                    # segmentation_predictions.append(segmentation_long_half_2)
    
                    # count_all = count_all + torch.sum(torch.abs(label_half1), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half1)+label_half1)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half1)-label_half1)/2, dim=0)
                    # count_all = count_all + torch.sum(torch.abs(label_half2), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half2)+label_half2)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half2)-label_half2)/2, dim=0)
    
                    batch_time.update(time.time() - end)
                    end = time.time()
    
                    desc = f'Test (spot.): '
                    desc += f'Time {batch_time.avg:.3f}s '
                    desc += f'(it:{batch_time.val:.3f}s) '
                    desc += f'Data:{data_time.avg:.3f}s '
                    desc += f'(it:{data_time.val:.3f}s) '
                    t.set_description(desc)
    
    
    
                    def get_spot_from_NMS(Input, window, thresh=0.0):
    
                        detections_tmp = np.copy(Input)
                        # res = np.empty(np.size(Input), dtype=bool)
                        indexes = []
                        MaxValues = []
                        while(np.max(detections_tmp) >= thresh):
    
                            # Get the max remaining index and value
                            max_value = np.max(detections_tmp)
                            max_index = np.argmax(detections_tmp)
                            MaxValues.append(max_value)
                            indexes.append(max_index)
                            # detections_NMS[max_index,i] = max_value
    
                            nms_from = int(np.maximum(-(window/2)+max_index,0))
                            nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                            detections_tmp[nms_from:nms_to] = -1
    
                        return np.transpose([indexes, MaxValues])
                    
                    
    
                    framerate = dataloader.dataset.framerate
                    get_spot = get_spot_from_BNMS2
    
                    json_data = dict()
                    json_data["UrlLocal"] = game_ID
                    json_data["predictions"] = list()
    
                    for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                        for l in range(dataloader.dataset.num_classes):
                            spots = get_spot(
                                timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold, min_window=3)
                            for spot in spots:
                                # print("spot", int(spot[0]), spot[1], spot)
                                frame_index = int(spot[0])
                                confidence = spot[1]
                                # confidence = predictions_half_1[frame_index, l]
    
                                seconds = int((frame_index//framerate)%60)
                                minutes = int((frame_index//framerate)//60)
    
                                prediction_data = dict()
                                prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                                if dataloader.dataset.version == 2:
                                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                                else:
                                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                                prediction_data["position"] = str(int((frame_index/framerate)*1000))
                                prediction_data["half"] = str(half+1)
                                prediction_data["confidence"] = str(confidence)
                                json_data["predictions"].append(prediction_data)
                    
                    os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                    with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                        json.dump(json_data, output_file, indent=4)
            else:

                for i, (game_ID, feat1_half1, feat2_half1, feat1_half2, feat2_half2, label_half1, label_half2) in t:

                    data_time.update(time.time() - end)
    
                    # Batch size of 1
                    
                    game_ID = game_ID[0]
                    feat1_half1 = feat1_half1.squeeze(0)
                    feat2_half1 = feat2_half1.squeeze(0)
                    label_half1 = label_half1.float().squeeze(0)
                    feat1_half2 = feat1_half2.squeeze(0)
                    feat2_half2 = feat2_half2.squeeze(0)
                    label_half2 = label_half2.float().squeeze(0)
                    
    
                    # Compute the output for batches of frames
                    BS = 256
                    timestamp_long_half_1 = []
                    for b in range(int(np.ceil(len(feat1_half1)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat1_half1) else len(feat1_half1)-1
                        feat1 = feat1_half1[start_frame:end_frame].cuda()
                        feat2 = feat2_half1[start_frame:end_frame].cuda()
                        output, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feat1, feat2)
                        output = output.cpu().detach().numpy()
                        timestamp_long_half_1.append(output)
                    timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)
    
                    timestamp_long_half_2 = []
                    for b in range(int(np.ceil(len(feat1_half2)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat1_half2) else len(feat1_half2)-1
                        feat1 = feat1_half2[start_frame:end_frame].cuda()
                        feat2 = feat2_half2[start_frame:end_frame].cuda()
                        output, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feat1, feat2)
                        output = output.cpu().detach().numpy()
                        timestamp_long_half_2.append(output)
                    timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)
    
    
                    timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                    timestamp_long_half_2 = timestamp_long_half_2[:, 1:]
    
                    spotting_grountruth.append(torch.abs(label_half1))
                    spotting_grountruth.append(torch.abs(label_half2))
                    spotting_grountruth_visibility.append(label_half1)
                    spotting_grountruth_visibility.append(label_half2)
                    spotting_predictions.append(timestamp_long_half_1)
                    spotting_predictions.append(timestamp_long_half_2)
                    # segmentation_predictions.append(segmentation_long_half_1)
                    # segmentation_predictions.append(segmentation_long_half_2)
    
                    # count_all = count_all + torch.sum(torch.abs(label_half1), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half1)+label_half1)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half1)-label_half1)/2, dim=0)
                    # count_all = count_all + torch.sum(torch.abs(label_half2), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half2)+label_half2)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half2)-label_half2)/2, dim=0)
    
                    batch_time.update(time.time() - end)
                    end = time.time()
    
                    desc = f'Test (spot.): '
                    desc += f'Time {batch_time.avg:.3f}s '
                    desc += f'(it:{batch_time.val:.3f}s) '
                    desc += f'Data:{data_time.avg:.3f}s '
                    desc += f'(it:{data_time.val:.3f}s) '
                    t.set_description(desc)
    
    
    
                    def get_spot_from_NMS(Input, window, thresh=0.0, min_window=3):
    
                        detections_tmp = np.copy(Input)
                        # res = np.empty(np.size(Input), dtype=bool)
                        indexes = []
                        MaxValues = []
                        while(np.max(detections_tmp) >= thresh):
    
                            # Get the max remaining index and value
                            max_value = np.max(detections_tmp)
                            max_index = np.argmax(detections_tmp)
                            
                            # detections_NMS[max_index,i] = max_value
    
                            nms_from = int(np.maximum(-(window/2)+max_index,0))
                            nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                            
                            if (detections_tmp[nms_from:nms_to] >= thresh).sum() > min_window:
                                MaxValues.append(max_value)
                                indexes.append(max_index)
                            detections_tmp[nms_from:nms_to] = -1
    
                        return np.transpose([indexes, MaxValues])
                    
                    def get_spot_from_BNMS(Input, window, thresh= 0.0):
                        detections_tmp = np.copy(Input)
                        indexes = []
                        MaxValues = []
                        while(np.max(detections_tmp) >= thresh):
                            max_value = np.max(detections_tmp)
                            max_index = np.argmax(detections_tmp)
                            MaxValues.append(max_value)
        
                            nms_from = int(np.maximum(-(window/2)+max_index,0))
                            nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                                                       
                            best_index = (np.arange(nms_from, nms_to) * detections_tmp[nms_from:nms_to]).sum() / detections_tmp[nms_from:nms_to].sum()
                            
                            indexes.append(round(best_index))
                            detections_tmp[nms_from:nms_to] = 0
                        return np.transpose([indexes, MaxValues])
                    
                    def get_spot_from_BNMS2(Input, window, thresh= 0.0, min_window = 0, max_window = 20):
                        detections_tmp = np.copy(Input)
                        not_indexes = ([i for i, x in enumerate(detections_tmp < thresh) if x])
                        if len(not_indexes) == 0:
                            not_indexes = [0, len(detections_tmp)]
                        indexes = []
                        MaxValues = []
                        for i in range(len(not_indexes)-1):
                            action_length = not_indexes[i+1] - not_indexes[i] -1
                            if (action_length) > (min_window):
                                if action_length > max_window:
                                    splits = (math.ceil(action_length / max_window))
                                    for j in range(splits):
                                        nms_from = not_indexes[i]+1+j*max_window
                                        nms_to = np.minimum(not_indexes[i]+(j+1)*max_window, len(detections_tmp))
                                        max_value = np.mean(detections_tmp[nms_from:nms_to])
                                        best_index = (np.arange(nms_from, nms_to) * detections_tmp[nms_from:nms_to]).sum() / detections_tmp[nms_from:nms_to].sum()
                                        indexes.append(round(best_index))
                                        MaxValues.append(max_value)
                                else:
                                    nms_from = not_indexes[i] +1
                                    nms_to = not_indexes[i+1]
                                    max_value = np.mean(detections_tmp[nms_from:nms_to])
                                    best_index = (np.arange(nms_from, nms_to) * detections_tmp[nms_from:nms_to]).sum() / detections_tmp[nms_from:nms_to].sum()
                                    indexes.append(round(best_index))
                                    MaxValues.append(max_value)
            
                        return np.transpose([indexes, MaxValues])
    
                    framerate = dataloader.dataset.framerate
                    get_spot = get_spot_from_NMS
    
                    json_data = dict()
                    json_data["UrlLocal"] = game_ID
                    json_data["predictions"] = list()
                    nms_window = [12, 7, 20, 9, 9, 9, 9, 7, 7, 7, 7, 7, 20, 20, 9, 20, 20]
                    for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                        
                        for l in range(dataloader.dataset.num_classes):
                            spots = get_spot(
                                timestamp[:, l], window=nms_window[l]*framerate, thresh=NMS_threshold, min_window = 0)

                            for spot in spots:
                                # print("spot", int(spot[0]), spot[1], spot)
                                frame_index = int(spot[0])
                                confidence = spot[1]
                                # confidence = predictions_half_1[frame_index, l]
    
                                seconds = int((frame_index//framerate)%60)
                                minutes = int((frame_index//framerate)//60)
    
                                prediction_data = dict()
                                prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                                if dataloader.dataset.version == 2:
                                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                                else:
                                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                                prediction_data["position"] = str(int((frame_index/framerate)*1000))
                                prediction_data["half"] = str(half+1)
                                prediction_data["confidence"] = str(confidence)
                                json_data["predictions"].append(prediction_data)
                    
                    os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                    with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                        json.dump(json_data, output_file, indent=4)



        def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])


        # zip folder
        zipResults(zip_path=output_results,
                target_dir = os.path.join("models", model_name, output_folder),
                filename="results_spotting.json")

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
    results = evaluate(SoccerNet_path=labels_path, 
                 Predictions_path=output_results,
                 split="test",
                 prediction_file="results_spotting.json", 
                 version=dataloader.dataset.version,
                 metric="tight")    

    
    return results

    # return a_mAP
  
def testSpottingEnsemble(path, model_name, split, overwrite=True, NMS_window=30, NMS_threshold=0.2, ensemble_method = 'mean', ensemble_chunk = 3):

    split2 = '_'.join([split])
    chunk_sizes = [2, 3, 4, 5]#, 7]
    
    # print(split)
    output_results = os.path.join("models", model_name, f"results_spotting_{split2}.zip")
    output_folder = f"outputs_{split2}"

    if not os.path.exists(output_results) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        framerate = 1

        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        timestamps_long_half_1 = []
        timestamps_long_half_2 = []
        game_IDs = []
        if split == 'valid':
            split_aux = ['train', 'valid', 'test']
        else:
            split_aux = [split]
        
        for chunk_size in chunk_sizes:
            dataset_Test  = SoccerNetClipsTesting(path='Baidu+ResNet', features='baidu_soccer_embeddings.npy', split=split_aux, version=2, framerate=1, chunk_size=chunk_size*1)
            print('Test loader')
            dataloader = torch.utils.data.DataLoader(dataset_Test,
                batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True)
            
            model = Model(weights=None, input_size=8576,
                          num_classes=dataloader.dataset.num_classes, chunk_size=chunk_size*1,
                          framerate=1, pool='final_model').cuda()
            checkpoint = torch.load(os.path.join("models", 'Pooling', 'model_chunk' + str(chunk_size) + '_full.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()



            #end = time.time()
            with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
                #Spotting if not Baidu+ResNet
    
                for i, (game_ID, feat1_half1, feat2_half1, feat1_half2, feat2_half2, label_half1, label_half2) in t:
    
                    #data_time.update(time.time() - end)
        
                    # Batch size of 1
                        
                    game_ID = game_ID[0]
                    feat1_half1 = feat1_half1.squeeze(0)
                    feat2_half1 = feat2_half1.squeeze(0)
                    label_half1 = label_half1.float().squeeze(0)
                    feat1_half2 = feat1_half2.squeeze(0)
                    feat2_half2 = feat2_half2.squeeze(0)
                    label_half2 = label_half2.float().squeeze(0)
                    
        
                    # Compute the output for batches of frames
                    BS = 256
                    timestamp_long_half_1 = []
                    for b in range(int(np.ceil(len(feat1_half1)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat1_half1) else len(feat1_half1)-1
                        feat1 = feat1_half1[start_frame:end_frame].cuda()
                        feat2 = feat2_half1[start_frame:end_frame].cuda()
                        output, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feat1, feat2)
                        output = output.cpu().detach().numpy()
                        timestamp_long_half_1.append(output)
                    timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)
        
                    timestamp_long_half_2 = []
                    for b in range(int(np.ceil(len(feat1_half2)/BS))):
                        start_frame = BS*b
                        end_frame = BS*(b+1) if BS * \
                            (b+1) < len(feat1_half2) else len(feat1_half2)-1
                        feat1 = feat1_half2[start_frame:end_frame].cuda()
                        feat2 = feat2_half2[start_frame:end_frame].cuda()
                        output, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5 = model(feat1, feat2)
                        output = output.cpu().detach().numpy()
                        timestamp_long_half_2.append(output)
                    timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)
        
        
                    timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                    timestamp_long_half_2 = timestamp_long_half_2[:, 1:]
                    
                    
                    
        
                    spotting_grountruth.append(torch.abs(label_half1))
                    spotting_grountruth.append(torch.abs(label_half2))
                    spotting_grountruth_visibility.append(label_half1)
                    spotting_grountruth_visibility.append(label_half2)
                    spotting_predictions.append(timestamp_long_half_1)
                    spotting_predictions.append(timestamp_long_half_2)
                    
                    timestamps_long_half_1.append(timestamp_long_half_1)
                    timestamps_long_half_2.append(timestamp_long_half_2)
                    game_IDs.append(game_ID)
                    # segmentation_predictions.append(segmentation_long_half_1)
                    # segmentation_predictions.append(segmentation_long_half_2)
        
                    # count_all = count_all + torch.sum(torch.abs(label_half1), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half1)+label_half1)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half1)-label_half1)/2, dim=0)
                    # count_all = count_all + torch.sum(torch.abs(label_half2), dim=0)
                    # count_visible = count_visible + torch.sum((torch.abs(label_half2)+label_half2)/2, dim=0)
                    # count_unshown = count_unshown + torch.sum((torch.abs(label_half2)-label_half2)/2, dim=0)
    
                    #batch_time.update(time.time() - end)
                    #end = time.time()
        
                    desc = f'Test (spot.): '
                    desc += f'Time {batch_time.avg:.3f}s '
                    desc += f'(it:{batch_time.val:.3f}s) '
                    desc += f'Data:{data_time.avg:.3f}s '
                    desc += f'(it:{data_time.val:.3f}s) '
                    t.set_description(desc)
    
    
        #Ensemble
        n_matches = int(len(timestamps_long_half_1)/len(chunk_sizes))
        print('Ensembling...')
        
        def get_spot_from_NMS(Input, window, thresh=0.0, min_window=3):

            detections_tmp = np.copy(Input)
            # res = np.empty(np.size(Input), dtype=bool)
            indexes = []
            MaxValues = []
            while(np.max(detections_tmp) >= thresh):

                # Get the max remaining index and value
                max_value = np.max(detections_tmp)
                max_index = np.argmax(detections_tmp)
                
                # detections_NMS[max_index,i] = max_value

                nms_from = int(np.maximum(-(window/2)+max_index,0))
                nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                
                if (detections_tmp[nms_from:nms_to] >= thresh).sum() > min_window:
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                detections_tmp[nms_from:nms_to] = -1

            return np.transpose([indexes, MaxValues])
        
        if split != 'valid':
            if ensemble_method == 'MLP':
                #Load trained model
                model = EnsembleModel(ensemble_chunk = ensemble_chunk, n_models = len(chunk_sizes)).cuda()
                checkpoint = torch.load(os.path.join("models", 'ensemble', "model.pth.tar"))
                model.load_state_dict(checkpoint['state_dict'])
            
            for m in range(n_matches):
                if ensemble_method == 'best_model_class':
                    best_models =np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 5])
                    timestamp_long_half_1 = np.zeros(timestamps_long_half_1[m].shape)
                    timestamp_long_half_2 = np.zeros(timestamps_long_half_2[m].shape)
                    for j in range(len(chunk_sizes)):
                        chunk = chunk_sizes[j]
                        ids = (best_models == chunk)
                        timestamp_long_half_1[:, ids] = timestamps_long_half_1[m + n_matches * j][:, ids]
                        timestamp_long_half_2[:, ids] = timestamps_long_half_2[m + n_matches * j][:, ids]
                
                if ensemble_method == 'mean':
                    timestamp_long_half_1 = np.zeros(timestamps_long_half_1[m].shape)
                    timestamp_long_half_2 = np.zeros(timestamps_long_half_2[m].shape)
                    for j in range(len(chunk_sizes)):
                        timestamp_long_half_1 += timestamps_long_half_1[m + n_matches * j]
                        timestamp_long_half_2 += timestamps_long_half_2[m + n_matches * j]
                    timestamp_long_half_1 /= len(chunk_sizes)
                    timestamp_long_half_2 /= len(chunk_sizes)
                    
                if ensemble_method == 'weighted_mean':
                    model_weights = np.array([0.541, 0.5728, 0.5187, 0.5351, 0.3966]) ** 2
                    timestamp_long_half_1 = np.zeros(timestamps_long_half_1[m].shape)
                    timestamp_long_half_2 = np.zeros(timestamps_long_half_2[m].shape)
                    for j in range(len(chunk_sizes)):
                        timestamp_long_half_1 += timestamps_long_half_1[m + n_matches * j] * model_weights[j]
                        timestamp_long_half_2 += timestamps_long_half_2[m + n_matches * j] * model_weights[j]
                    timestamp_long_half_1 /= model_weights.sum()
                    timestamp_long_half_2 /= model_weights.sum()
                    
                if ensemble_method == 'weighted_mean2':
                    model_weights = np.array([[0.6893, 0.5455, 0.6962, 0.4447, 0.4519, 0.5189, 0.5235, 0.5239, 0.6988, 0.6722, 0.6562, 0.4086, 0.6142, 0.7429, 0.5034, 0.3122, 0.1953],
                                              [0.8025, 0.5726, 0.769, 0.4757, 0.5294, 0.5333, 0.5274, 0.5395, 0.7392, 0.6961, 0.6911, 0.4366, 0.5886, 0.7683, 0.5133, 0.2147, 0.3402],
                                              [0.744, 0.4997, 0.6725, 0.4095, 0.4516, 0.506, 0.4838, 0.5045, 0.6659, 0.6378, 0.6214, 0.4095, 0.5995, 0.708, 0.4909, 0.2534, 0.1606],
                                              [0.7458, 0.5091, 0.6985, 0.4164, 0.4635, 0.4959, 0.5116, 0.506, 0.6967, 0.6583, 0.6439, 0.4152, 0.5255, 0.6845, 0.5068, 0.276, 0.3434]
                                              ]) ** 2
                    timestamp_long_half_1 = np.zeros(timestamps_long_half_1[m].shape)
                    timestamp_long_half_2 = np.zeros(timestamps_long_half_2[m].shape)
                    for j in range(len(chunk_sizes)):
                        timestamp_long_half_1 += timestamps_long_half_1[m + n_matches * j] * model_weights[j]
                        timestamp_long_half_2 += timestamps_long_half_2[m + n_matches * j] * model_weights[j]
                    timestamp_long_half_1 /= model_weights.sum(axis = 0)
                    timestamp_long_half_2 /= model_weights.sum(axis = 0)
                    
                if ensemble_method == 'MLP':
                    for j in range(len(chunk_sizes)):
                        if j == 0:
                            full_preds1 = timestamps_long_half_1[m + n_matches * j]
                            full_preds2 = timestamps_long_half_2[m + n_matches * j]
                        else:
                            full_preds1 = np.concatenate((full_preds1, timestamps_long_half_1[m + n_matches * j]), axis = 1)
                            full_preds2 = np.concatenate((full_preds2, timestamps_long_half_2[m + n_matches * j]), axis = 1)
                    
                    full_preds1 = torch.from_numpy(feats2clip(full_preds1, 1, ensemble_chunk, off=int(ensemble_chunk/2))).cuda()
                    full_preds2 = torch.from_numpy(feats2clip(full_preds2, 1, ensemble_chunk, off=int(ensemble_chunk/2))).cuda()
                    output1 = []
                    for j in range(0, math.ceil((full_preds1.shape[0] - 1) // 100) + 1):
                        output1.append(model(full_preds1[(j * 100): np.minimum((j+1) * 100, full_preds1.shape[0]), :, :]).cpu().detach().numpy())
                    output2 = []
                    for j in range(0, math.ceil((full_preds2.shape[0] - 1) // 100) + 1):
                        output2.append(model(full_preds2[(j * 100): np.minimum((j+1) * 100, full_preds2.shape[0]), :, :]).cpu().detach().numpy())
                    
                    timestamp_long_half_1 = np.concatenate(output1)
                    timestamp_long_half_2 = np.concatenate(output2)
                    if m == 0:
                        print(timestamp_long_half_1.shape)
                        print(timestamp_long_half_1)
                        print(timestamp_long_half_2.shape)
                        print(timestamp_long_half_2)
                    
                framerate = dataloader.dataset.framerate
                get_spot = get_spot_from_NMS
        
                json_data = dict()
                json_data["UrlLocal"] = game_ID
                json_data["predictions"] = list()
                nms_window = [12, 7, 20, 9, 9, 9, 9, 7, 7, 7, 7, 7, 20, 20, 9, 20, 20]
                #nms_window = 7
                for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                            
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(
                            timestamp[:, l], window=nms_window[l]*framerate, thresh=NMS_threshold, min_window = 0)
                        #spots = get_spot(
                        #    timestamp[:, l], window=nms_window*framerate, thresh=NMS_threshold, min_window = 0)
                        
                        for spot in spots:
                            # print("spot", int(spot[0]), spot[1], spot)
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            # confidence = predictions_half_1[frame_index, l]
    
                            seconds = int((frame_index//framerate)%60)
                            minutes = int((frame_index//framerate)//60)
    
                            prediction_data = dict()
                            prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                            if dataloader.dataset.version == 2:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                            else:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                            prediction_data["position"] = str(int((frame_index/framerate)*1000))
                            prediction_data["half"] = str(half+1)
                            prediction_data["confidence"] = str(confidence)
                            json_data["predictions"].append(prediction_data)
                    
                os.makedirs(os.path.join("models", model_name, output_folder, game_IDs[m]), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_IDs[m], "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

###### FINS AQUÍ

        else:
            
            dict_event = EVENT_DICTIONARY_V2
            path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
            all_preds = []
            all_labels = []
            for m in range(n_matches):
                for j in range(len(chunk_sizes)):
                    if j == 0:
                        full_preds1 = timestamps_long_half_1[m + n_matches * j]
                        full_preds2 = timestamps_long_half_2[m + n_matches * j]
                    else:
                        full_preds1 = np.concatenate((full_preds1, timestamps_long_half_1[m + n_matches * j]), axis = 1)
                        full_preds2 = np.concatenate((full_preds2, timestamps_long_half_2[m + n_matches * j]), axis = 1)
                
                full_preds1 = feats2clip(full_preds1, 1, ensemble_chunk, off=int(ensemble_chunk/2))
                full_preds2 = feats2clip(full_preds2, 1, ensemble_chunk, off=int(ensemble_chunk/2))
                all_preds.append(full_preds1)
                all_preds.append(full_preds2)
                
                #Get labels of each frame for match
                labels = json.load(open(os.path.join(path_labels, game_IDs[m], "Labels-v2.json")))
                label_half1 = np.zeros((full_preds1.shape[0], 17))
                label_half2 = np.zeros((full_preds2.shape[0], 17))
                
                for annotation in labels["annotations"]:
    
                    time = annotation["gameTime"]
                    event = annotation["label"]
    
                    half = int(time[0])
    
                    minutes = int(time[-5:-3])
                    seconds = int(time[-2::])
                    frame = framerate * ( seconds + 60 * minutes ) 
    
                    if event not in dict_event:
                        continue
                    label = dict_event[event]
    
                    # if label outside temporal of view
                    if half == 1 and frame>=label_half1.shape[0]:
                        continue
                    if half == 2 and frame>=label_half2.shape[0]:
                        continue
                        
                    if half == 1:
                        #label_half1[max(0, frame-3)][label] = 0.5
                        #label_half1[max(0, frame-2)][label] = 0.75
                        #label_half1[max(0, frame-1)][label] = 0.9
                        #label_half1[min(label_half1.shape[0]-1, frame+3)][label] = 0.5
                        #label_half1[min(label_half1.shape[0]-1, frame+2)][label] = 0.75
                        #label_half1[min(label_half1.shape[0]-1, frame+1)][label] = 0.9                        
                        label_half1[frame][label] = 1
                        
    
                    if half == 2:
                        #label_half2[max(0, frame-3)][label] = 0.5
                        #label_half2[max(0, frame-2)][label] = 0.75
                        #label_half2[max(0, frame-1)][label] = 0.9
                        #label_half2[min(label_half2.shape[0]-1, frame+3)][label] = 0.5
                        #label_half2[min(label_half2.shape[0]-1, frame+2)][label] = 0.75
                        #label_half2[min(label_half2.shape[0]-1, frame+1)][label] = 0.9
                        label_half2[frame][label] = 1

                all_labels.append(label_half1)
                all_labels.append(label_half2)
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            
            #idx1 = np.arange(0, all_labels.shape[0])#[(1 - (all_labels.sum(axis = 1) == 0) * random.choices([0, 1], weights = [0.9, 0.1], k = len(all_labels))).astype('bool')]
            #print(idx1)
            #idx2 = np.arange(0, labels_half2.shape[0])[(1 - (labels_half2[:, 0] == 1) * random.choices([0, 1], weights = [0.9, 0.1], k = len(labels_half2))).astype('bool')]
            #train_prob = 0.8
            #true_false_split = (np.random.uniform(size = all_labels.shape[0]) > train_prob)
            #idx_train = np.arange(0, all_labels.shape[0])[(1 - true_false_split).astype('bool')]
            #idx_test = np.arange(0, all_labels.shape[0])[true_false_split]
            
            
            
            dataset_train_ensemble = TrainEnsemble(all_preds, all_labels)
            dataset_val_ensemble = TrainEnsemble(all_preds, all_labels)
            train_loader = torch.utils.data.DataLoader(dataset_train_ensemble,
                batch_size=512,
                num_workers=1, shuffle = True, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(dataset_val_ensemble,
                batch_size=512,
                num_workers=1, shuffle = False, pin_memory=True)
            model = EnsembleModel(ensemble_chunk = ensemble_chunk, n_models = len(chunk_sizes)).cuda()
            logging.info(model)
            total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
            parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
            logging.info("Total number of parameters: " + str(total_params))
            
            criterion = NLLLoss_weights_ensemble()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, 
                                        betas=(0.9, 0.999), eps=1e-08, 
                                        weight_decay=1e-5, amsgrad=True)
            
                        
            # start training
            trainer('ensemble', train_loader, val_loader, train_loader, 
                    model, optimizer, criterion, patience=1000,
                    model_name='ensemble',
                    max_epochs=50, evaluation_frequency=10000)
            
            print('asdf')
            
            return 0



        def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])


        # zip folder
        zipResults(zip_path=output_results,
                target_dir = os.path.join("models", model_name, output_folder),
                filename="results_spotting.json")

    if split2 == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
    results = evaluate(SoccerNet_path=labels_path, 
                 Predictions_path=output_results,
                 split="test",
                 prediction_file="results_spotting.json", 
                 version=dataloader.dataset.version,
                 metric="tight")    

    
    return results

