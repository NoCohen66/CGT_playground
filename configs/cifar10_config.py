import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

means = (0.4914, 0.4822, 0.4465)
stddevs = (0.2470, 0.2435, 0.2616) 
img_shape = (3, 32, 32)
num_classes = 10

def get_dataset(augment=True, get_train=True, get_val=False):
    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        )
    else:
        train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    
    if get_train:
        train = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=True, download=True, transform=train_transform)
        if get_val:
            train_size = int(len(train) * 0.8)
            indices = [k for k in range(len(train))]
            random.shuffle(indices)
            train = torch.utils.data.Subset(train, indices[:train_size])
            val = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=True, download=False, transform=test_transform)
            val = torch.utils.data.Subset(val, indices[train_size:])
            return train, val
        else:
            return train
    else:
        test = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=False, download=True, transform=test_transform)
        return test

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(4096, 150)
        self.fc2 = nn.Linear(150, 10)
    
    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


gtss_train = [
    {
        ROTATE: (-10, 10, 0.001)
    },
    {
        ROTATE: (-2, 2, 0.02),
        SHEAR: (-0.02, 0.02, 0.0005)
    },
    {
        SCALE: (0.99, 1.01, 0.00005),
        ROTATE: (-1, 1, 0.005),
        CONTRAST: (0.99, 1.01, 0.005),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]

gtss_certify = [
    {
        ROTATE: (-10, 10, 0.0002)
    },
    {
        ROTATE: (-2, 2, 0.01),
        SHEAR: (-0.02, 0.02, 0.00025)
    },
    {
        SCALE: (0.99, 1.01, 0.00005),
        ROTATE: (-1, 1, 0.005),
        CONTRAST: (0.99, 1.01, 0.005),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]

# ABLATION CONFIGS
# gtss_train = [
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.015),
#         SHEAR: (-0.02, 0.02, 0.000375)
#     },
#     {
#         ROTATE: (-2, 2, 0.025),
#         SHEAR: (-0.02, 0.02, 0.000625)
#     },
#     {
#         ROTATE: (-2, 2, 0.03),
#         SHEAR: (-0.02, 0.02, 0.00075)
#     },
# ]

# gtss_certify = [
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
# ]
