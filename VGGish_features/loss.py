'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file defines the loss function for the VGGish model
'''

import torch

#Define loss function for audio
class NLLLoss_audio(torch.nn.Module):
    def __init__(self):
        super(NLLLoss_audio, self).__init__()
        #self.weights1 = weights1
        
    def forward(self, labels, output):
        weights = torch.tensor([1, 20, 10, 15, 10, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 30, 30, 50]).cuda()
        #weights = torch.tensor([1, 40, 20, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 80, 80]).cuda()
        return torch.mean(torch.mean(weights * labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0))