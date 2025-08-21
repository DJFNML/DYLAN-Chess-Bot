import torch, torchmetrics, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy
from torch.nn import CrossEntropyLoss

from ChessDataset import ChessDataset
from TrainTestSplit import TrainTestSplit as TTS


trainData, testData = TTS("./Documents/mygames.csv", "ngsterAHH")
TrainSet = ChessDataset(trainData)

dataloader_train = DataLoader(
    TrainSet,
    batch_size=2,
    shuffle=True,
)

features, labels = next(iter(dataloader_train))



