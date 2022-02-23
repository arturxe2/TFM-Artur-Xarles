import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


'''
Function to define my loss (not working right now)
'''
def my_loss(y_true, y_pred):
    #y_true to float 
    y_true = tf.cast(y_true, tf.float32)
   #Loss function (introducing epsilons to avoid 0)
    loss1 = tf.reduce_mean(y_true * -tf.math.log(tf.math.maximum(y_pred, tf.keras.backend.epsilon())) + (1-y_true) * (-tf.math.log(tf.math.maximum(1 - y_pred, tf.keras.backend.epsilon()))))
    loss = tf.reduce_mean(loss1)
    return loss
    

'''
Function to read data to train the model, with size chunk * n_features. It also
returns the outputs in a one-hot-encoding for all the possible classes. Allow to 
choose a window_size to decide the frequence of the samples and the dataset that 
you want to use (train in general)
'''
def read_data(chunks = 60, data_split = "train", window_size = 60):
    #Initialize values
    i = 0
    n_total = 0
    y_train = ['Elements']
    X = []
    #Define train matches directories
    path = '/data-net/datasets/SoccerNetv2/data_split/'
    init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'
    with open(path + data_split + '.txt') as f:
        lines = f.readlines()
    #Read data for each match
    for line in lines:
        i += 1
        #1 -> 1st half, 2 -> 2nd half
        #Load .npy files
        features1 = np.load(init_path + line.rstrip('\n') + '/1_ResNET_TF2.npy')
        features2 = np.load(init_path + line.rstrip('\n') + '/2_ResNET_TF2.npy')
        #Define number of chunks second splits
        n_frames1 = features1.shape[0]
        n_frames2 = features2.shape[0]
        #Read actions for match and determine the nº of frame it corresponds
        actions = pd.DataFrame(json.load(open(init_path + line.rstrip('\n') + '/Labels-v2.json'))['annotations'])
        actions['half'] = actions['gameTime'].apply(lambda x: int(x[0]))
        actions['minute'] = actions['gameTime'].apply(lambda x: x[4:])
        actions['frame'] = actions['minute'].apply(lambda x: int(x[0:2]) * 60 * 2 + int(x[3:5]) * 2)
        #Split data in 60-frames chunks (for 1st half)
        for n in range((n_frames1 - chunks) // window_size):
            #Collect features
            x = features1[(n * window_size) : (n * window_size + chunks), :]
            X.append(x.tolist())
            #Collect outputs
            y = actions['label'][(actions['frame'] >= n * window_size) & (actions['frame'] < (n * window_size + chunks)) & (actions['half'] == 1)].values
            if len(y) == 0:
                y = ['Background']
            else:
                y = y.tolist()
            y_train.append(y)
            n_total += 1
            
        #Split data in 60-frames chunks (for 2nd half)
        for n in range((n_frames2 - chunks) // window_size):
            #Collect features
            x = features2[(n * window_size) : (n * window_size + chunks), :]
            X.append(x.tolist())
            #Collect outputs
            y = actions['label'][(actions['frame'] >= n * window_size) & (actions['frame'] < (n * window_size + chunks)) & (actions['half'] == 2)].values
            
            if len(y) == 0:
                y = ['Background']
            else:
                y = y.tolist()
            y_train.append(y)
            n_total += 1
    
        #Print the number of the match we are
        print('Data collected for ' + str(i) + ' matches.')
        if i == 10:
            break
    
    #Resize data, and put output in one-hot-encoding
    print('Resizing features...')
    X = np.array(X).reshape(n_total, chunks, features1.shape[1])
    print('Getting output one-hot encoding')
    y_train = y_train[1:]
    aux = pd.Series(y_train)
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(aux), columns = mlb.classes_, index = aux.index)
    classes = res.columns
    res = np.array(res)

    return X, res, classes

'''
Function to create a basic model. Takes the x_train, applies a MaxPooling in 
the temporal dimension and applies it to a FCNN. Outputs a vector of sigmoid neurons
'''
def max_pooling(x_train, y_train):
    #Input and output sizes
    chunks = x_train.shape[1]
    input_shape = (chunks, x_train.shape[2])
    output_shape = y_train.shape[1]
    #Define model
    model = keras.Sequential(
        [
            keras.Input(shape = input_shape),
            layers.MaxPooling1D(pool_size = chunks),
            layers.Flatten(),
            layers.Dense(x_train.shape[2] * 2, activation = 'relu'),
            layers.Dropout(0.6),
            layers.Dense(output_shape, activation = "sigmoid"),
            ]
        )
    print(model.summary)
    #Compile model
    model.compile(loss = "BinaryCrossentropy", optimizer = "Adam", metrics = ["Accuracy", "Precision"])
    #Train model
    model.fit(x_train, y_train, epochs = 10, validation_split = 0.2)
    return model


'''
Function that given a model that outputs a one-hot-encoding vector of predictions
for classes and for a chunk, returns an array with as many rows as frames of the match
and as many columns as classes, with the probability for each frame to have the 
class action
'''
def make_predictions(model, n_classes, line, chunks = 60, data_split = "test", frames_window = 2):
    #Pathes of the data
    init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'
    print(line)
    #1 -> 1st half, 2 -> 2nd half
    #Load .npy files
    features1 = np.load(init_path + line.rstrip('\n') + '/1_ResNET_TF2.npy')
    features2 = np.load(init_path + line.rstrip('\n') + '/2_ResNET_TF2.npy')
    n_frames1 = features1.shape[0]
    n_frames2 = features2.shape[0]
    #Initialize array of probabilities for each class and frame
    action_frame1 = np.zeros((n_frames1, n_classes))
    action_frame2 = np.zeros((n_frames2, n_classes))
    #Initialize number of predictions made for each frame
    n_preds1 = np.zeros(n_frames1)
    n_preds2 = np.zeros(n_frames2)
    #Predict 1st half actions
    print('Predicting 1st half actions...')
    for x in range((n_frames1 - chunks) // frames_window):
        action_frame1[(x * frames_window) : (x * frames_window + chunks), :] += (model.predict(features1[(x * frames_window) : (x * frames_window + chunks), :].reshape(1, chunks, features1.shape[1])))
        n_preds1[(x * frames_window): (x * frames_window + chunks)] += 1
    action_frame1[(n_frames1 - chunks):(n_frames1), :]+= model.predict(features1[(n_frames1 - chunks):(n_frames1), :].reshape(1, chunks, features1.shape[1]))
    n_preds1[(n_frames1 - chunks):(n_frames1)] += 1
    #Predict 2nd half actions
    print('Predicting 2nd half actions...')
    for x in range((n_frames2 - chunks) // frames_window):
        action_frame2[(x * frames_window) : (x * frames_window + chunks), :] += (model.predict(features2[(x * frames_window) : (x * frames_window + chunks), :].reshape(1, chunks, features2.shape[1])))
        n_preds2[(x * frames_window): (x * frames_window + chunks)] += 1
    action_frame2[(n_frames2 - chunks):(n_frames2), :]+= model.predict(features2[(n_frames2 - chunks):(n_frames2), :].reshape(1, chunks, features2.shape[1]))
    n_preds2[(n_frames2 - chunks):(n_frames2)] += 1
    #Normalize
    action_frame1 = action_frame1 / n_preds1[:, None]
    action_frame2 = action_frame2 / n_preds2[:, None]        

    return action_frame1, action_frame2


'''
Given the array of probabilities of class for frame applies NMS to select the 
moment of an event given a treshold and the number of comparisons around it
'''
def NMS_spotting(action_frame, n_comparisons = 10, treshold = 0.4):
    frames, n_classes = action_frame.shape
    for i in range(frames):
        max_bool = (action_frame[i, :] == (action_frame[max(0, i - n_comparisons):(i + 1), :].max(axis = 0))) & (action_frame[i, :] >= (np.ones(n_classes) * treshold))
        action_frame[i, :] = (max_bool * action_frame[i, :])
        if i > 0:
            action_frame[max(0, i - n_comparisons): i, :] = (1 - max_bool)[None, :] * action_frame[max(0, i - n_comparisons): i, :]
    #action_frame = (action_frame > 0).astype("float")
    return action_frame


'''
Function that given the spots per frame for 2 halfs of a match returns a dictionary
with the desired structure for spotting evaluation.
'''
def prediction_output(spots1, spots2, labels, match):
    positions1, positions2 = spots1.nonzero()
    sol = {}
    half = 1
    action = []
    for comb in zip(positions1, positions2):
        minu = comb[0] // (2 * 60)
        sec = comb[0] % (2 * 60) / 2
    
        dict = {"gameTime": str(half) + " - " + str(minu) + ":" + str("%02d" % sec), 
               "label": labels[comb[1]],
               "position": (minu * 60 + sec) * 1000, 
               "half": half,
               "confidence": spots1[comb[0], comb[1]]}
        action.append(dict)
    positions1, positions2 = spots2.nonzero()
    half = 2
    for comb in zip(positions1, positions2):
        minu = comb[0] // (2 * 60)
        sec = comb[0] % (2 * 60) / 2
    
        dict = {"gameTime": str(half) + " - " + str(minu) + ":" + str("%02d" % sec), 
               "label": labels[comb[1]],
               "position": (minu * 60 + sec) * 1000, 
               "half": half,
               "confidence": spots2[comb[0], comb[1]]}
        action.append(dict)
    sol.update({"UrlLocal": match, "predictions": action})
    
    return(sol)
       
'''
MAIN CODE
'''

#Define parameters:
chunks = 120
window_size_class = chunks
window_size_pred = 40
n_comparisons_NMS = 15

#Methods used:
x_train, y_train, classes = read_data(chunks = chunks, data_split = "train", window_size = window_size_class)
n_classes = y_train.shape[1]
model = max_pooling(x_train, y_train)

#Iteration to make predictions for each test match:
#Pathes of the data
path = '/data-net/datasets/SoccerNetv2/data_split/'

with open(path + 'test.txt') as f:
    lines = f.readlines()
for line in lines:
#For each match
#Predictions for a match
    preds1, preds2 = make_predictions(model = model, n_classes = n_classes, line = line, chunks = chunks, data_split = "test", frames_window = window_size_pred)
#Spotting for each half
    spots1 = NMS_spotting(preds1, n_comparisons = n_comparisons_NMS)
    spots2 = NMS_spotting(preds2, n_comparisons = n_comparisons_NMS)
#Dictionary output
    solution = prediction_output(spots1, spots2, classes)
    print(solution)
#Return some interesting things:
    print(spots1.sum(axis = 0))
    print(spots2.sum(axis = 0))
    print(classes)


'''
Extra code to save data:
np.save('/home-net/axesparraguera/data/x_train.npy', x_train)
np.save('/home-net/axesparraguera/data/y_train.npy', y_train)

x_train = np.load('/home-net/axesparraguera/data/x_train.npy')
y_train = np.load('/home-net/axesparraguera/data/y_train.npy')

classes = ['Background', 'Ball out of play', 'Clearance', 'Corner', 'Direct free-kick', 
           'Foul', 'Goal', 'Indirect free-kick', 'Kick-off', 'Offside', 'Penalty', 'Red card', 
           'Shots off target', 'Shots on target', 'Substitution', 'Throw-in', 'Yellow card', 
           'Yellow->red card']
'''
