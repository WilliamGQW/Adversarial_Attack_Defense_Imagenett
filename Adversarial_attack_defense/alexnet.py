import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

# The general structure was refered from the original alexnet paper 
# http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf

class AlexNet_2(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_2, self).__init__()
        # 5 Conv layers with Relu and maxpool
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),      
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 384 * 13 * 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256 * 13 * 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 256 * 6 * 6
        )
        # 3 fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    # forward pass
    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.linear(x)
        return x