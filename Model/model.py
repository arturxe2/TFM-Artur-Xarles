'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file defines the HMTAS model and the model for the fusion learning ensemble
'''
import __future__
import numpy as np
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, add: float = 0.) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] + add
        return self.dropout(x)


#Model class
class Model(nn.Module):
    def __init__(self, weights=None, num_classes=3, chunk_size=240, framerate=2, model="HMTAS"):
        """
        INPUT: a Tensor of shape (batch_size,chunk_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Model, self).__init__()

        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.model = model


        
        
        if self.model == "HMTAS":
            #All features to 512 dimensionality
            #self.conv1A = nn.Conv1d(512, 512, 1, stride=1, bias=False)
            self.conv1A = nn.Conv1d(128, 512, 1, stride=1, bias=False)
            self.normA = nn.BatchNorm1d(512)
            self.conv1B = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.norm1B = nn.BatchNorm1d(512)
            self.conv2B = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.norm2B = nn.BatchNorm1d(512)
            self.conv3B = nn.Conv1d(384, 512, 1, stride=1, bias=False)
            self.norm3B = nn.BatchNorm1d(512)
            self.conv4B = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.norm4B = nn.BatchNorm1d(512)
            self.conv5B = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.norm5B = nn.BatchNorm1d(512)
            
            self.relu = nn.ReLU()
            
            #Encoders for each feature vector
            encoder_layerA = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderA = nn.TransformerEncoder(encoder_layerA, 1) 
            encoder_layerB1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB1 = nn.TransformerEncoder(encoder_layerB1, 1)
            encoder_layerB2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB2 = nn.TransformerEncoder(encoder_layerB2, 1)
            encoder_layerB3 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB3 = nn.TransformerEncoder(encoder_layerB3, 1)
            encoder_layerB4 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB4 = nn.TransformerEncoder(encoder_layerB4, 1)
            encoder_layerB5 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB5 = nn.TransformerEncoder(encoder_layerB5, 1)
            
            #Segons encoders
            
            encoder_layerA_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderA_2 = nn.TransformerEncoder(encoder_layerA_2, 1) 
            encoder_layerB1_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB1_2 = nn.TransformerEncoder(encoder_layerB1_2, 1)
            encoder_layerB2_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB2_2 = nn.TransformerEncoder(encoder_layerB2_2, 1)
            encoder_layerB3_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB3_2 = nn.TransformerEncoder(encoder_layerB3_2, 1)
            encoder_layerB4_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB4_2 = nn.TransformerEncoder(encoder_layerB4_2, 1)
            encoder_layerB5_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoderB5_2 = nn.TransformerEncoder(encoder_layerB5_2, 1)
            
            
            
            #Reduce for each feature vector to 18
            self.pool_layerA = nn.MaxPool1d(chunk_size * 1, stride = 1)
            self.fcA = nn.Linear(512, self.num_classes+1)
            self.pool_layerB1 = nn.MaxPool1d(chunk_size, stride=1)
            self.fcB1 = nn.Linear(512, self.num_classes+1)
            self.pool_layerB2 = nn.MaxPool1d(chunk_size, stride=1)
            self.fcB2 = nn.Linear(512, self.num_classes+1)
            self.pool_layerB3 = nn.MaxPool1d(chunk_size, stride=1)
            self.fcB3 = nn.Linear(512, self.num_classes+1)
            self.pool_layerB4 = nn.MaxPool1d(chunk_size, stride=1)
            self.fcB4 = nn.Linear(512, self.num_classes+1)
            self.pool_layerB5 = nn.MaxPool1d(chunk_size, stride=1)
            self.fcB5 = nn.Linear(512, self.num_classes+1)
            
            #Transformer mix
            encoder_layer_mix = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoder_mix = nn.TransformerEncoder(encoder_layer_mix, 1)
            encoder_layer_mix_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoder_mix_2 = nn.TransformerEncoder(encoder_layer_mix_2, 1)
            
            #Reduce mix to 18
            #self.pool_layer_mix = nn.MaxPool1d(chunk_size*(2+1*5), stride=1)
            self.pool_layer_mix = nn.MaxPool1d(6, stride = 1)
            self.fc_mix = nn.Linear(512, self.num_classes+1)
            




        self.drop2 = nn.Dropout(p=0.1)
        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))
    #def forward(self, inputs):
    def forward(self, inputs1, inputs2):
        

        if self.model == "HMTAS":
            inputsA = inputs2.float()
            inputsB = inputs1.float()
            
            inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size * 2)
            inputsB = inputsB.permute((0, 2, 1)) #(B x n_features x chunk_size)
            inputsB1 = inputsB[:, :2048, :]
            inputsB2 = inputsB[:, 2048:4096, :]
            inputsB3 = inputsB[:, 4096:4480, :]
            inputsB4 = inputsB[:, 4480:6528, :]
            inputsB5 = inputsB[:, 6528:, :]
            
            #Reduce to 256 dimensionality each
            inputsA = self.relu(self.normA(self.conv1A((inputsA)))) #(B x 256 x chunk_size * 2)
            inputsB1 = self.relu(self.norm1B(self.conv1B(inputsB1))) #(B x 256 x chunk_size)
            inputsB2 = self.relu(self.norm2B(self.conv2B(inputsB2))) #(B x 256 x chunk_size)
            inputsB3 = self.relu(self.norm3B(self.conv3B(inputsB3))) #(B x 256 x chunk_size)
            inputsB4 = self.relu(self.norm4B(self.conv4B(inputsB4))) #(B x 256 x chunk_size)
            inputsB5 = self.relu(self.norm5B(self.conv5B(inputsB5))) #(B x 256 x chunk_size)
            
            inputsA = inputsA.permute((0, 2, 1))
            inputsB1 = inputsB1.permute((0, 2, 1))
            inputsB2 = inputsB2.permute((0, 2, 1))
            inputsB3 = inputsB3.permute((0, 2, 1))
            inputsB4 = inputsB4.permute((0, 2, 1))
            inputsB5 = inputsB5.permute((0, 2, 1))
            
            #Transformers 1
            inputsA = self.encoderA(self.drop(inputsA))
            inputsB1 = self.encoderB1(self.drop(inputsB1))
            inputsB2 = self.encoderB2(self.drop(inputsB2))
            inputsB3 = self.encoderB3(self.drop(inputsB3))
            inputsB4 = self.encoderB4(self.drop(inputsB4))
            inputsB5 = self.encoderB5(self.drop(inputsB5))
            
            #Transformers 2
            
            inputsA = self.encoderA_2(inputsA)
            inputsB1 = self.encoderB1_2(inputsB1)
            inputsB2 = self.encoderB2_2(inputsB2)
            inputsB3 = self.encoderB3_2(inputsB3)
            inputsB4 = self.encoderB4_2(inputsB4)
            inputsB5 = self.encoderB5_2(inputsB5)
            
            inputsA = inputsA.permute((0, 2, 1))
            inputsB1 = inputsB1.permute((0, 2, 1))
            inputsB2 = inputsB2.permute((0, 2, 1))
            inputsB3 = inputsB3.permute((0, 2, 1))
            inputsB4 = inputsB4.permute((0, 2, 1))
            inputsB5 = inputsB5.permute((0, 2, 1))
            
            inputs_pooledA = self.pool_layerA(inputsA) #(B x 256 x 1)
            inputs_pooledB1 = self.pool_layerB1(inputsB1) #(B x 256 x1)
            inputs_pooledB2 = self.pool_layerB2(inputsB2) #(B x 256 x 1)
            inputs_pooledB3 = self.pool_layerB3(inputsB3) #(B x 256 x 1)
            inputs_pooledB4 = self.pool_layerB4(inputsB4) #(B x 256 x 1)
            inputs_pooledB5 = self.pool_layerB5(inputsB5) #(B x 256 x 1)
            
            inputs_pooledA_out = inputs_pooledA.squeeze(-1) #(B x 256)
            inputs_pooledB1_out = inputs_pooledB1.squeeze(-1) #(B x 256)
            inputs_pooledB2_out = inputs_pooledB2.squeeze(-1) #(B x 256)
            inputs_pooledB3_out = inputs_pooledB3.squeeze(-1) #(B x 256)
            inputs_pooledB4_out = inputs_pooledB4.squeeze(-1) #(B x 256)
            inputs_pooledB5_out = inputs_pooledB5.squeeze(-1) #(B x 256)
            
            inputsA = inputs_pooledA.permute((0, 2, 1)) #(B x 1 x 256)
            inputsB1 = inputs_pooledB1.permute((0, 2, 1)) #(B x 1 x 256)
            inputsB2 = inputs_pooledB2.permute((0, 2, 1)) #(B x 1 x 256)
            inputsB3 = inputs_pooledB3.permute((0, 2, 1)) #(B x 1 x 256)
            inputsB4 = inputs_pooledB4.permute((0, 2, 1)) #(B x 1 x 256)
            inputsB5 = inputs_pooledB5.permute((0, 2, 1)) #(B x 1 x 256)
            
            
            inputs_mix = torch.cat((inputsA, inputsB1, inputsB2, inputsB3, inputsB4, inputsB5), dim=1)
            
            
            #Transformer mix (B x chunk_size * 7 x 256)
            #inputs_mix = inputs_mix.permute((0, 2, 1))
            #inputs_mix = self.pool_layer_mix(inputs_mix)
            #inputs_mix = inputs_mix.permute((0, 2, 1))
            
            inputs_mix = self.encoder_mix(self.drop(inputs_mix)) 
            inputs_mix = self.encoder_mix_2(inputs_mix)
            
            
            
            inputs_mix = inputs_mix.permute((0, 2, 1))
                                                    
            #Individual outputs
            
            
            
            
            inputs_pooled_mix = self.pool_layer_mix(inputs_mix).squeeze(-1) #(B x 256)
            
            outputsA = self.sigm(self.fcA(self.drop(inputs_pooledA_out)))
            outputsB1 = self.sigm(self.fcB1(self.drop(inputs_pooledB1_out)))
            outputsB2 = self.sigm(self.fcB2(self.drop(inputs_pooledB2_out)))
            outputsB3 = self.sigm(self.fcB3(self.drop(inputs_pooledB3_out)))
            outputsB4 = self.sigm(self.fcB4(self.drop(inputs_pooledB4_out)))
            outputsB5 = self.sigm(self.fcB5(self.drop(inputs_pooledB5_out)))
            
            outputs_mix = self.sigm(self.fc_mix(self.drop(inputs_pooled_mix)))
            
        return outputs_mix, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5
            
        
    

#Define Ensemble model experiments
class EnsembleModel(nn.Module):

    def __init__(self, ensemble_chunk = 3, n_models = 2):
        super(EnsembleModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_models * 17, nhead=n_models)
        self.pool_layer = nn.MaxPool1d(ensemble_chunk, stride = 1)
        self.encoder = nn.TransformerEncoder(encoder_layer, 1) 
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=n_models*17, nhead=n_models)
        self.encoder2 = nn.TransformerEncoder(encoder_layer2, 1)
        encoder_layer3 = nn.TransformerEncoderLayer(d_model=n_models*17, nhead=n_models)
        self.encoder3 = nn.TransformerEncoder(encoder_layer3, 1)
        self.fc = nn.Linear(n_models*17, 17)
        self.fc2 = nn.Linear(n_models*17, 17)
        self.fc3 = nn.Linear(n_models * 17, 17)
        self.drop = nn.Dropout(p=0.3)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # Input B x 3 x 34
        inputs = inputs.float()
        inputs = self.encoder((inputs))
        inputs = self.encoder2((inputs))
        inputs = self.encoder3((inputs))
        inputs = inputs.permute((0, 2, 1))
        inputs = self.pool_layer(inputs)
        inputs = inputs.squeeze(-1)
        #outputs = self.relu(self.fc((inputs)))
        outputs = self.sigm(self.fc((inputs)))
        #outputs = self.sigm(self.fc3((outputs)))
        

        
        return outputs
