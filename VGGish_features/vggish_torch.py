import torch
from torchsummary import summary

print('asdfasdf')
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()
print(model)
print(summary(model, (100, 96, 64)))