
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
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, framerate=2, pool="NetVLAD"):
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
            
        elif self.pool == "transformer_2features_2":
            self.conv1R = nn.Conv1d(512, 512, 1, stride=1, bias=False)
            self.normR = nn.BatchNorm1d(512)
            self.conv1B = nn.Conv1d(8576, 512, 1, stride=1, bias=False)
            self.normB = nn.BatchNorm1d(512)
            self.relu = nn.ReLU()
            #Add segment embedding
            self.pos_encoder = PositionalEncoding(512, )
            
            #Baidu encoders
            encoder_layer1B = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder1B = nn.TransformerEncoder(encoder_layer1B, 1)
            
            encoder_layer2B = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder2B = nn.TransformerEncoder(encoder_layer2B, 1)
            
            encoder_layer3B = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder3B = nn.TransformerEncoder(encoder_layer3B, 1)
            
            #Audio encoders
            encoder_layer1R = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder1R = nn.TransformerEncoder(encoder_layer1R, 1)
            
            encoder_layer2R = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder2R = nn.TransformerEncoder(encoder_layer2R, 1)
            
            encoder_layer3R = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=64)
            self.encoder3R = nn.TransformerEncoder(encoder_layer3R, 1)
            #Pool layer
            self.pool_layerB = nn.MaxPool1d(chunk_size * (1), stride=1)
            self.pool_layerR = nn.MaxPool1d(chunk_size * 2, stride = 1)
            self.fc = nn.Linear(1024, self.num_classes+1)
        
        elif self.pool == "transformer_2features_3":
            self.conv1R = nn.Conv1d(512, 512, 1, stride=1, bias=False)
            self.normR = nn.BatchNorm1d(512)
            self.conv1B1 = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.normB1 = nn.BatchNorm1d(512)
            self.conv1B2 = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.normB2 = nn.BatchNorm1d(512)
            self.conv1B3 = nn.Conv1d(384, 512, 1, stride=1, bias=False)
            self.normB3 = nn.BatchNorm1d(512)
            self.conv1B4 = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.normB4 = nn.BatchNorm1d(512)
            self.conv1B5 = nn.Conv1d(2048, 512, 1, stride=1, bias=False)
            self.normB5 = nn.BatchNorm1d(512)
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
            self.pool_layer = nn.MaxPool1d(chunk_size * (2 + 5), stride=1)
            self.fc2 = nn.Linear(512, self.num_classes+1)

        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(num_clusters=64, dim=512,
                                      normalize_input=True, vladv2=False)
            self.fc = nn.Linear(input_size*64, self.num_classes+1)

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

        elif self.pool == "transformer_2features_2":
            inputs1 = inputs1.float()
            inputs2 = inputs2.float()
            
            inputs1 = self.drop(inputs1)
            inputs2 = self.drop(inputs2)
            
            inputsB = inputs1.permute((0, 2, 1))
            inputsR = inputs2.permute((0, 2, 1))
            inputsR = self.relu(self.normR(self.conv1R(inputsR)))#(B x 512 x (chunk_size * 2))
            inputsB = self.relu(self.normB(self.conv1B(inputsB))) #(B x 512 x (chunk_size))
            inputsR = inputsR.permute((0, 2, 1))#(B x (chunk_size * 2) x 512)
            inputsB = inputsB.permute((0, 2, 1))#(B x (chunk_size) x 512)
            
            #Positional encodding + feature encoding (1 for R, 0 for B)
            inputsR = self.pos_encoder(inputsR, add=0.)#(B x (chunk_size * 2) x 512)
            inputsB = self.pos_encoder(inputsB, add=0.)#(B x (chunk_size) x 512)

            inputsR = self.encoder1R(inputsR)
            inputsR = self.encoder2R(inputsR)
            inputsR = self.encoder3R(inputsR)
            
            inputsB = self.encoder1B(inputsB)
            inputsB = self.encoder2B(inputsB)
            inputsB = self.encoder3B(inputsB)
            
            inputsR = inputsR.permute((0, 2, 1))
            inputsB = inputsB.permute((0, 2, 1))
            
            inputsR_pooled = self.pool_layerR(inputsR)
            inputsB_pooled = self.pool_layerB(inputsB)
            
            inputsR_pooled = inputsR_pooled.squeeze(-1)
            inputsB_pooled = inputsB_pooled.squeeze(-1)
            
            inputs = torch.cat((inputsR_pooled, inputsB_pooled), dim=1)
            
            outputs = self.sigm(self.fc(self.drop(inputs)))
            
        elif self.pool == "transformer_2features_3":
            inputs1 = inputs1.float()
            inputs2 = inputs2.float()
            inputsB = inputs1.permute((0, 2, 1))
            inputsR = inputs2.permute((0, 2, 1))
            inputsR = self.relu(self.normR(self.conv1R(inputsR)))#(B x 512 x (chunk_size * 2))
            inputsB1 = inputsB[:, :2048, :]
            inputsB2 = inputsB[:, 2048:4096, :]
            inputsB3 = inputsB[:, 4096:4480, :]
            inputsB4 = inputsB[:, 4480:6528, :]
            inputsB5 = inputsB[:, 6528:8576, :]
            inputsB1 = self.relu(self.normB1(self.conv1B1(inputsB1))) #(B x 512 x (chunk_size))
            inputsB2 = self.relu(self.normB2(self.conv1B2(inputsB2))) #(B x 512 x (chunk_size))
            inputsB3 = self.relu(self.normB3(self.conv1B3(inputsB3))) #(B x 512 x (chunk_size))
            inputsB4 = self.relu(self.normB4(self.conv1B4(inputsB4))) #(B x 512 x (chunk_size))
            inputsB5 = self.relu(self.normB5(self.conv1B5(inputsB5))) #(B x 512 x (chunk_size))
            inputsR = inputsR.permute((0, 2, 1))#(B x (chunk_size * 2) x 512)
            inputsB1 = inputsB1.permute((0, 2, 1))#(B x (chunk_size) x 512)
            inputsB2 = inputsB1.permute((0, 2, 1))#(B x (chunk_size) x 512)
            inputsB3 = inputsB1.permute((0, 2, 1))#(B x (chunk_size) x 512)
            inputsB4 = inputsB1.permute((0, 2, 1))#(B x (chunk_size) x 512)
            inputsB5 = inputsB1.permute((0, 2, 1))#(B x (chunk_size) x 512)
            
            #Positional encodding + feature encoding (1 for R, 0 for B)
            inputsR = self.pos_encoder(inputsR, add=0.)#(B x (chunk_size * 2) x 512)
            inputsB1 = self.pos_encoder(inputsB1, add=1.)#(B x (chunk_size) x 512)
            inputsB2 = self.pos_encoder(inputsB2, add=2.)#(B x (chunk_size) x 512)
            inputsB3 = self.pos_encoder(inputsB3, add=3.)#(B x (chunk_size) x 512)
            inputsB4 = self.pos_encoder(inputsB4, add=4.)#(B x (chunk_size) x 512)
            inputsB5 = self.pos_encoder(inputsB5, add=5.)#(B x (chunk_size) x 512)

            
            inputs = torch.cat((inputsR, inputsB1, inputsB2, inputsB3, inputsB4, inputsB5), dim=1) #(B x (chunk_size * (1 + 2)) x 512)
            #Encoders
            inputs = self.encoder(inputs) #(B x (chunk_size * (1 + 2)) x 512)
            inputs = self.encoder2(inputs)
            inputs = self.encoder3(inputs)
            
            inputs = inputs.permute((0, 2, 1))
            inputs_pooled = self.pool_layer(inputs)
            inputs_pooled = inputs_pooled.squeeze(-1)
            outputs = self.sigm(self.fc2(self.drop(inputs_pooled)))
            
            
            

        elif self.pool == "NetVLAD":
            inputs = inputs.unsqueeze(-1)
            inputs = inputs.permute((0, 2, 1, 3))
            inputs_pooled = self.pool_layer(inputs)

        # Extra FC layer with dropout and sigmoid activation
        #output = self.sigm(self.fc2(self.drop(inputs_pooled)))

        return outputs
