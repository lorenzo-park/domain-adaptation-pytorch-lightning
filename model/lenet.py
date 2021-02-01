from model.component import Flatten, DebugLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
        )
        self.classifier = nn.Sequential(
#             DebugLayer(),
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=10),
#             DebugLayer(),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=2),
        )
