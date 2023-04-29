import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import random

from .layers import *
import sys
sys.path.append('../')
from definitions import *

means = 0.1307
stddevs = 0.3081
img_shape = (1, 28, 28)
num_classes = 10

def get_dataset(get_train=True, get_val=False):
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    
    if get_train:
        train = torchvision.datasets.MNIST(root='datasets/MNIST_Dataset', train=True, download=True, transform=train_transform)
        if get_val:
            train_size = int(len(train) * 0.8)
            indices = [k for k in range(len(train))]
            random.shuffle(indices)
            train = torch.utils.data.Subset(train, indices[:train_size])
            val = torchvision.datasets.MNIST(root='datasets/MNIST_Dataset', train=True, download=False, transform=test_transform)
            val = torch.utils.data.Subset(val, indices[train_size:])
            return train, val
        else:
            return train
    else:
        test = torchvision.datasets.MNIST(root='datasets/MNIST_Dataset', train=False, download=True, transform=test_transform)
        return test

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(3136, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


gtss_train = [
    {
        ROTATE: (-30, 30, 0.5)
    },
    {
        TRANSLATE_X: (-2, 2, 0.1),
        TRANSLATE_Y: (-2, 2, 0.1)
    },
    {
        SCALE: (0.95, 1.05, 0.01),
        ROTATE: (-5, 5, 0.25),
        CONTRAST: (0.95, 1.05, 0.05),
        BRIGHTNESS: (-0.01, 0.01, 0.02)
    },
    {
        SHEAR: (-0.02, 0.02, 0.005),
        ROTATE: (-2, 2, 0.125),
        SCALE: (0.98, 1.02, 0.005),
        CONTRAST: (0.98, 1.02, 0.04),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]

gtss_certify = [
    {
        ROTATE: (-30, 30, 0.25)
    },
    {
        TRANSLATE_X: (-2, 2, 0.05),
        TRANSLATE_Y: (-2, 2, 0.05)
    },
    {
        SCALE: (0.95, 1.05, 0.005),
        ROTATE: (-5, 5, 0.125),
        CONTRAST: (0.95, 1.05, 0.05),
        BRIGHTNESS: (-0.01, 0.01, 0.02)
    },
    {
        SHEAR: (-0.02, 0.02, 0.0025),
        ROTATE: (-2, 2, 0.0625),
        SCALE: (0.98, 1.02, 0.0025),
        CONTRAST: (0.98, 1.02, 0.04),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]
