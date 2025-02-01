import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Grayscale
import os

# 1*28*28
def MNIST_dataset(batch_size, test_batch_size):
    
    train_set = datasets.MNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.MNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    
    return train_set, test_set, train_loader, test_loader
