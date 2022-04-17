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
        #weights = torch.tensor([1, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5]).cuda()
        return torch.mean(torch.mean(4 * labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0))
