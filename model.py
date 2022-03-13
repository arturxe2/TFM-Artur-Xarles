
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.pos_encoder = PositionalEncoding(512, )
            self.encoder = nn.TransformerEncoder(encoder_layer, 1) 
            self.pool_layer = nn.MaxPool1d(chunk_size, stride=1)
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

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)

        # Temporal pooling operation
        if self.pool == "MAX":
            inputs = inputs.permute((0, 2, 1))
            inputs_pooled = self.pool_layer(inputs)
            inputs_pooled = inputs_pooled.squeeze(-1)
            
        elif self.pool == "MAX512":
            inputs = inputs.permute((0, 2, 1))
            inputs = self.relu(self.norm(self.conv1(inputs)))
            #breakpoint()
            inputs_pooled = self.pool_layer(inputs)
            #breakpoint()
            inputs_pooled = inputs_pooled.squeeze(-1)
            #breakpoint()
            #### Transformer
        elif self.pool == "MAX512_transformer":
            inputs = inputs.permute((0, 2, 1))
            print(inputs.shape)
            inputs = self.relu(self.norm(self.conv1(inputs)))
            print(inputs.shape)
            inputs = inputs.permute((0, 2, 1))
            print(inputs.shape)
            inputs = self.pos_encoder(inputs)
            print(inputs.shape)
            inputs = self.encoder(inputs)
            print(inputs.shape)
            #breakpoint()
            inputs = inputs.permute((0, 2, 1))
            print(inputs.shape)
            inputs_pooled = self.pool_layer(inputs)
            print(inputs_pooled.shape)
            #breakpoint()
            inputs_pooled = inputs_pooled.squeeze(-1)
            #breakpoint()
            #### Transformer

        elif self.pool == "NetVLAD":
            inputs = inputs.unsqueeze(-1)
            inputs = inputs.permute((0, 2, 1, 3))
            inputs_pooled = self.pool_layer(inputs)

        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc2(self.drop(inputs_pooled)))

        return output
