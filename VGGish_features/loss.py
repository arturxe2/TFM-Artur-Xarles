import torch


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))
    
class NLLLoss_weights(torch.nn.Module):
    def __init__(self):
        super(NLLLoss_weights, self).__init__()
        #self.weights1 = weights1
        
    def forward(self, labels, output):
        weights = torch.tensor([1, 20, 10, 15, 10, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 15, 30, 50]).cuda()
        #weights = torch.tensor([1, 40, 20, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 80, 80]).cuda()
        return torch.mean(torch.mean(weights * labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0))

class NLLLoss_weights2(torch.nn.Module):
    def __init__(self):
        super(NLLLoss_weights, self).__init__()
        #self.weights1 = weights1
        
    def forward(self, labels, output):
        weights = torch.tensor([1, 20, 10, 15, 10, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 30, 30, 50]).cuda()
        #weights = torch.tensor([1, 40, 20, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 80, 80]).cuda()
        return torch.mean(torch.mean(weights * labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0))