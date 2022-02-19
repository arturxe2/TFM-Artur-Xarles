import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers

def read_data_train(chunks = 60):
    #Initialize values
    i = 0
    n_total = 0
    y_train = ['Elements']
    X = []
    #Define train matches directories
    path = '/data-net/datasets/SoccerNetv2/data_split/'
    init_path = '/data-net/datasets/SoccerNetv2/ResNET_TF2/'
    with open(path + 'train.txt') as f:
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
        if i == 200:
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
            layers.Dense(200, activation = 'relu'),
            layers.Dropout(0.4),
            layers.Dense(output_shape, activation = "sigmoid"),
            ]
        )
    print(model.summary)
    #Compile model
    model.compile(loss = "BinaryCrossentropy", optimizer = "Adam", metrics = ["Accuracy", "Precision"])
    #Train model
    model.fit(x_train, y_train, epochs = 300, validation_split = 0.2)
    
    return model
    
chunks = 120
x_train, y_train, classes = read_data_train(chunks = chunks)
model = max_pooling(x_train, y_train)

print(np.round(model.predict(x_train[0:5, :, :]), 2))
print(y_train[0:5])
print(classes)
