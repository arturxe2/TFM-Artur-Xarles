
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

from random import shuffle

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

import vggish_input
import vggish_params
import vggish_slim
from vggish_torch import *

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



flags = tf.app.flags

flags.DEFINE_integer(
    'num_batches', 30,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 18

class SoccerNetClips(Dataset):
    def __init__(self, path="/data-net/datasets/SoccerNetv2/videos_lowres", features="audio.npy", labels="labels.npy", 
                 split=["train", "valid", "test"], version=2, val_split = 0.8):
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
        indexes = np.random.rand(self.n)
        self.train_indexes = np.arange(0, self.n)[indexes <= val_split]
        self.val_indexes = np.arange(0, self.n)[indexes > val_split]
        
        
    def __get_sample__(self, n_samples):

        indexes = np.random.choice(self.train_indexes, size = n_samples)

        return self.game_feats[indexes, :, :], self.game_labels[indexes, :]
    
    def __get_val__(self):
        return self.game_feats[self.val_indexes, :, :], self.game_labels[self.val_indexes, :]
        
        






def main(_):
    
  
    '''
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(training=FLAGS.train_vggish)
    saver = tf.train.Saver()
    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units. Add an activation function
      # to the embeddings since they are pre-activation.
      num_units = 100
      fc = slim.fully_connected(tf.nn.relu(embeddings), num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(
          fc, _NUM_CLASSES, activation_fn=None, scope='logits')
      tf.sigmoid(logits, name='prediction')

      # Add training ops.
      with tf.variable_scope('train'):
        global_step = tf.train.create_global_step()

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels_input = tf.placeholder(
            tf.float32, shape=(None, _NUM_CLASSES), name='labels')

        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels_input, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=vggish_params.LEARNING_RATE,
            epsilon=vggish_params.ADAM_EPSILON)
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # The training loop.
    features_input = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    for _ in range(FLAGS.num_batches):
      (features_train, labels_train) = a.__get_sample__(50)
      (features_val, labels_val) = a.__get_val__()
      [num_steps, loss_value, _] = sess.run(
          [global_step, loss, train_op],
          feed_dict={features_input: features_train, labels_input: labels_train})
      loss_val = sess.run(loss, feed_dict={features_input: features_val, labels_input: labels_val})
      print('Step %d: loss %g' % (num_steps, loss_value))
      print('Step %d: val loss %g' % (num_steps, loss_val))
    save_path = saver.save(sess, 'fine_tunned_vggish.ckpt')
    print('Saved finetunned model in : ' + save_path)
    
    '''

if __name__ == '__main__':

    model = get_vggish(with_classifier=True, pretrained=True)
    model.classifier._modules['2'] = nn.Linear(100, 18)
    print(model)
    