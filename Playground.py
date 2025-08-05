import torch, torchmetrics, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy
from torch.nn import CrossEntropyLoss
x = torch.rand(10,10)
print(x)
