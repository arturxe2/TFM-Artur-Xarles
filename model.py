
import __future__

import numpy as np
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import NetVLAD

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

class Model(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, vocab_size=128, chunk_size=240, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,chunk_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Model, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(chunk_size, stride=1)
            self.fc2 = nn.Linear(input_size, self.num_classes+1)
        
        elif self.pool == "MAX512":
            self.conv1 = nn.Conv1d(input_size, 512, 1, stride=1, bias=False)
            self.norm = nn.BatchNorm1d(512)
            self.relu = nn.ReLU()
            self.pool_layer = nn.MaxPool1d(chunk_size, stride=1)
            self.fc2 = nn.Linear(512, self.num_classes+1)
            
        elif self.pool == "MAX512_transformer":
            self.conv1 = nn.Conv1d(input_size, 512, 1, stride=1, bias=False)
            self.norm = nn.BatchNorm1d(512)
            self.relu = nn.ReLU()
            #First TransformerEncoder (1 layer)
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.pos_encoder = PositionalEncoding(512, )
            self.encoder = nn.TransformerEncoder(encoder_layer, 1) 
            
            #Second TransformerEncoder (1 layer)
            encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoder2 = nn.TransformerEncoder(encoder_layer2, 1)
            
            #Third TransformerEncoder
            encoder_layer3 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.encoder3 = nn.TransformerEncoder(encoder_layer3, 1)
            
            
            self.pool_layer = nn.MaxPool1d(chunk_size, stride=1)
            self.fc2 = nn.Linear(512, self.num_classes+1)
            
        elif self.pool == "transformer_2features":
            self.conv1R = nn.Conv1d(512, 512, 1, stride=1, bias=False)
            self.normR = nn.BatchNorm1d(512)
            self.conv1B = nn.Conv1d(8576, 512, 1, stride=1, bias=False)
            self.normB = nn.BatchNorm1d(512)
            self.relu = nn.ReLU()
            #Add segment embedding
            self.pos_encoder = PositionalEncoding(512, )
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder = nn.TransformerEncoder(encoder_layer, 1)
            
            encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder2 = nn.TransformerEncoder(encoder_layer2, 1)
            
            encoder_layer3 = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder3 = nn.TransformerEncoder(encoder_layer3, 1)
            #Pool layer
            self.pool_layer = nn.MaxPool1d(chunk_size * (2 + 1), stride=1)
            self.fc2 = nn.Linear(512, self.num_classes+1)
        
        elif self.pool == "final_model":
            #All features to 512 dimensionality
            self.conv1A = nn.Conv1d(512, 512, 1, stride=1, bias=False)
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
            self.pool_layerA = nn.MaxPool1d(chunk_size * 2, stride = 1)
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
            self.pool_layer_mix = nn.MaxPool1d(chunk_size*(2+1*5), stride=1)
            self.fc_mix = nn.Linear(512, self.num_classes+1)
            





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
        # input_shape: (batch,frames,dim_features)

        # Temporal pooling operation
        if self.pool == "MAX":
            inputs = inputs.permute((0, 2, 1))
            inputs_pooled = self.pool_layer(inputs)
            inputs_pooled = inputs_pooled.squeeze(-1)
            
        elif self.pool == "MAX512":
            inputs = inputs.float()
            inputs = inputs.permute((0, 2, 1))
            inputs = self.relu(self.norm(self.conv1(inputs)))
            #breakpoint()
            inputs_pooled = self.pool_layer(inputs)
            #breakpoint()
            inputs_pooled = inputs_pooled.squeeze(-1)
            #breakpoint()
            #### Transformer
        elif self.pool == "MAX512_transformer":
            inputs = inputs.float()
            inputs = inputs.permute((0, 2, 1)) #(B x n_features x n_frames)
            #print(inputs.shape)
            inputs = self.relu(self.norm(self.conv1(inputs))) #(B x 512 x n_frames)
            #print(inputs.shape)
            inputs = inputs.permute((0, 2, 1)) #(B x n_frames x 512)
            #print(inputs.shape)
            inputs = self.pos_encoder(inputs) #(B x n_frames x 512)
            #print(inputs.shape)
            inputs = self.encoder(inputs) #(B x n_frames x 512)
            #print(inputs.shape)
            #breakpoint()
            #inputs = self.pos_encoder2(inputs)
            inputs = self.encoder2(inputs)
            
            inputs = self.encoder3(inputs)
            inputs = inputs.permute((0, 2, 1)) #(B x 512 x n_frames)
            #print(inputs.shape)
            inputs_pooled = self.pool_layer(inputs) #(B x 512 x 1)
            #print(inputs_pooled.shape)
            #breakpoint()
            inputs_pooled = inputs_pooled.squeeze(-1)
            
            outputs = self.sigm(self.fc2(self.drop(inputs_pooled)))
            #breakpoint()
            #### Transformer
            
        elif self.pool == "transformer_2features":
            inputs1 = inputs1.float()
            inputs2 = inputs2.float()
            inputsB = inputs1.permute((0, 2, 1))
            inputsR = inputs2.permute((0, 2, 1))
            inputsR = self.relu(self.normR(self.conv1R(inputsR)))#(B x 512 x (chunk_size * 2))
            inputsB = self.relu(self.normB(self.conv1B(inputsB))) #(B x 512 x (chunk_size))
            inputsR = inputsR.permute((0, 2, 1))#(B x (chunk_size * 2) x 512)
            inputsB = inputsB.permute((0, 2, 1))#(B x (chunk_size) x 512)
            
            #Positional encodding + feature encoding (1 for R, 0 for B)
            inputsR = self.pos_encoder(inputsR, add=1.)#(B x (chunk_size * 2) x 512)
            inputsB = self.pos_encoder(inputsB, add=0.)#(B x (chunk_size) x 512)

            
            inputs = torch.cat((inputsB, inputsR), dim=1) #(B x (chunk_size * (1 + 2)) x 512)
            #Encoders
            inputs = self.encoder(inputs) #(B x (chunk_size * (1 + 2)) x 512)
            inputs = self.encoder2(inputs)
            inputs = self.encoder3(inputs)
            
            inputs = inputs.permute((0, 2, 1))
            inputs_pooled = self.pool_layer(inputs)
            inputs_pooled = inputs_pooled.squeeze(-1)
            
            outputs = self.sigm(self.fc2(self.drop(inputs_pooled)))

        elif self.pool == "final_model":
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
            inputsA = self.relu(self.normA(self.conv1A(inputsA))) #(B x 256 x chunk_size * 2)
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
            inputsA = self.encoderA(inputsA)
            inputsB1 = self.encoderB1(inputsB1)
            inputsB2 = self.encoderB2(inputsB2)
            inputsB3 = self.encoderB3(inputsB3)
            inputsB4 = self.encoderB4(inputsB4)
            inputsB5 = self.encoderB5(inputsB5)
            
            #Transformers 2
            
            inputsA = self.encoderA_2(inputsA)
            inputsB1 = self.encoderB1_2(inputsB1)
            inputsB2 = self.encoderB2_2(inputsB2)
            inputsB3 = self.encoderB3_2(inputsB3)
            inputsB4 = self.encoderB4_2(inputsB4)
            inputsB5 = self.encoderB5_2(inputsB5)
            
            
            inputs_mix = torch.cat((inputsA, inputsB1, inputsB2, inputsB3, inputsB4, inputsB5), dim=1)
            
            
            #Transformer mix
            inputs_mix = self.encoder_mix(inputs_mix) #(B x 256 x chunk_size * 7)
            inputs_mix = self.encoder_mix_2(inputs_mix)
            
            inputsA = inputsA.permute((0, 2, 1))
            inputsB1 = inputsB1.permute((0, 2, 1))
            inputsB2 = inputsB2.permute((0, 2, 1))
            inputsB3 = inputsB3.permute((0, 2, 1))
            inputsB4 = inputsB4.permute((0, 2, 1))
            inputsB5 = inputsB5.permute((0, 2, 1))
            
            inputs_mix = inputs_mix.permute((0, 2, 1))
                                                    
            #Individual outputs
            inputs_pooledA = self.pool_layerA(inputsA).squeeze(-1) #(B x 256 x 1)
            inputs_pooledB1 = self.pool_layerB1(inputsB1).squeeze(-1) #(B x 256)
            inputs_pooledB2 = self.pool_layerB2(inputsB2).squeeze(-1) #(B x 256)
            inputs_pooledB3 = self.pool_layerB3(inputsB3).squeeze(-1) #(B x 256)
            inputs_pooledB4 = self.pool_layerB4(inputsB4).squeeze(-1) #(B x 256)
            inputs_pooledB5 = self.pool_layerB5(inputsB5).squeeze(-1) #(B x 256)
            
            inputs_pooled_mix = self.pool_layer_mix(inputs_mix).squeeze(-1) #(B x 256)
            
            outputsA = self.sigm(self.fcA(self.drop(inputs_pooledA)))
            outputsB1 = self.sigm(self.fcB1(self.drop(inputs_pooledB1)))
            outputsB2 = self.sigm(self.fcB2(self.drop(inputs_pooledB2)))
            outputsB3 = self.sigm(self.fcB3(self.drop(inputs_pooledB3)))
            outputsB4 = self.sigm(self.fcB4(self.drop(inputs_pooledB4)))
            outputsB5 = self.sigm(self.fcB5(self.drop(inputs_pooledB5)))
            
            outputs_mix = self.sigm(self.fc_mix(self.drop(inputs_pooled_mix)))
            
            return outputs_mix, outputsA, outputsB1, outputsB2, outputsB3, outputsB4, outputsB5
            
        # Extra FC layer with dropout and sigmoid activation
        #output = self.sigm(self.fc2(self.drop(inputs_pooled)))

        return outputs
