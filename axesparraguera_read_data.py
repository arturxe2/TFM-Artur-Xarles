import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def my_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
   # tf.math.maximum(y_true, tf.keras.backend.epsilon())
    loss1 = tf.reduce_mean(y_true * -tf.math.log(tf.math.maximum(y_pred, tf.keras.backend.epsilon())) + (1-y_true) * (-tf.math.log(tf.math.maximum(1 - y_pred, tf.keras.backend.epsilon()))))
    print(loss1.get_shape())
    loss = tf.reduce_mean(loss1)
    return loss
    

def read_data(chunks = 60, data_split = "train"):
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
        n_chunks1 = int(features1.shape[0] // chunks)
        n_chunks2 = int(features2.shape[0] // chunks)
        n_total = n_total + n_chunks1 + n_chunks2
        #Read actions for match and determine the nº of frame it corresponds
        actions = pd.DataFrame(json.load(open(init_path + line.rstrip('\n') + '/Labels-v2.json'))['annotations'])
        actions['half'] = actions['gameTime'].apply(lambda x: int(x[0]))
        actions['minute'] = actions['gameTime'].apply(lambda x: x[4:])
        actions['frame'] = actions['minute'].apply(lambda x: int(x[0:2]) * 60 * 2 + int(x[3:5]) * 2)
        #Split data in 60-frames chunks (for 1st half)
        for n in range(n_chunks1):
            #Collect features
            x = features1[n * chunks : (n + 1) * chunks, :]
            X.append(x.tolist())
            #Collect outputs
            y = actions['label'][(actions['frame'] >= n * chunks) & (actions['frame'] < (n + 1) * chunks) & (actions['half'] == 1)].values
            if len(y) == 0:
                y = ['Background']
            else:
                y = y.tolist()
            y_train.append(y)
            
        #Split data in 60-frame chunks (for 2nd half)
        for n in range(n_chunks2):
            x = features2[n * chunks : (n + 1) * chunks, :]
            X.append(x.tolist())
            y = actions['label'][(actions['frame'] >= n * chunks) & (actions['frame'] < (n + 1) * chunks) & (actions['half'] == 2)].values
            if len(y) == 0:
                y = ['Background']
            else:
                y = y.tolist()
            y_train.append(y)
    
        #Print the number of the match we are
        print('Data collected for ' + str(i) + ' matches.')
        if i == 400:
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

def make_predictions(model, n_classes, chunks = 60, data_split = "test", frames_window = 2):
    i = 0
    path = '/data-net/datasets/SoccerNetv2/data_split/'
    init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'
    with open(path + data_split + '.txt') as f:
        lines = f.readlines()
    for line in lines:
        print(line)
        i += 1
        #1 -> 1st half, 2 -> 2nd half
        #Load .npy files
        features1 = np.load(init_path + line.rstrip('\n') + '/1_ResNET_TF2.npy')
        features2 = np.load(init_path + line.rstrip('\n') + '/2_ResNET_TF2.npy')
        n_frames1 = features1.shape[0]
        n_frames2 = features2.shape[0]
        action_frame1 = np.zeros((n_frames1, n_classes))
        action_frame2 = np.zeros((n_frames2, n_classes))
        n_preds1 = np.zeros(n_frames1)
        n_preds2 = np.zeros(n_frames2)
        print('Predicting 1st half actions...')
        for x in range((n_frames1 - chunks) // frames_window):
            action_frame1[(x * frames_window) : (x * frames_window + chunks), :] += (model.predict(features1[(x * frames_window) : (x * frames_window + chunks), :].reshape(1, chunks, features1.shape[1])))
            n_preds1[(x * frames_window): (x * frames_window + chunks)] += 1
        print('Predicting 2nd half actions...')
        for x in range((n_frames2 - chunks) // frames_window):
            action_frame2[(x * frames_window) : (x * frames_window + chunks), :] += (model.predict(features2[(x * frames_window) : (x * frames_window + chunks), :].reshape(1, chunks, features2.shape[1])))
            n_preds2[(x * frames_window): (x * frames_window + chunks)] += 1
        action_frame1 = action_frame1 / n_preds1[:, None]
        action_frame2 = action_frame2 / n_preds2[:, None]
        
        
        
        if i == 1:
            break

    return action_frame1, action_frame2

def spotting(action_frame, n_comparisons = 10, treshold = 0.4):
    frames, n_classes = action_frame.shape
    for i in range(frames):
        max_bool = (action_frame[i, :] == (action_frame[max(0, i - n_comparisons):(i + 1), :].max(axis = 0))) & (action_frame[i, :] >= (np.ones(n_classes) * treshold))
        action_frame[i, :] = (max_bool * action_frame[i, :])
        if i > 0:
            action_frame[max(0, i - n_comparisons): i, :] = (1 - max_bool)[None, :] * action_frame[max(0, i - n_comparisons): i, :]
    return action_frame
       
    
chunks = 120
#x_train, y_train, classes = read_data(chunks = chunks, data_split = "train")
#np.save('/home-net/axesparraguera/data/x_train.npy', x_train)
#np.save('/home-net/axesparraguera/data/y_train.npy', y_train)

x_train = np.load('/home-net/axesparraguera/data/x_train.npy')
y_train = np.load('/home-net/axesparraguera/data/y_train.npy')

classes = ['Background', 'Ball out of play', 'Clearance', 'Corner', 'Direct free-kick', 
           'Foul', 'Goal', 'Indirect free-kick', 'Kick-off', 'Offside', 'Penalty', 'Red card', 
           'Shots off target', 'Shots on target', 'Substitution', 'Throw-in', 'Yellow card', 
           'Yellow->red card']

#x_test, y_test, classes2 = read_data(chunks = chunks, data_split = "test")

#print(classes)
#print(classes2)
model = max_pooling(x_train, y_train)
n_classes = y_train.shape[1]
preds1, preds2 = make_predictions(model = model, n_classes = n_classes, chunks = chunks, data_split = "test", frames_window = 20)

spots = spotting(preds1)
print(spots)
print(spots.sum(axis = 0))

#print(preds1[0:30, :])
#print(preds2[0:30, :])
#print(model.evaluate(x_test, y_test))

#print(np.round(model.predict(x_train[0:20, :, :]), 2))
#print(y_train[0:20])
#print(classes)
#print(classes2)
