import torch


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))
    
class NLLLoss_weights(torch.nn.Module):
    def __initi__(self):
        super(NLLLoss_weights, self).__init__()
        
    def forward(self, labels, output):
        weights = torch.tensor([1.37, 100, 100, 100, 100, 100, 100, 100, 100, 12.8, 21.5, 34.5, 38.5, 100, 84.7, 100, 100, 100]).cuda()
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0) * weights)
