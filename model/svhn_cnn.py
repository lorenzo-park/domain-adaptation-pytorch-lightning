from model.component import Flatten, DebugLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
#             DebugLayer(),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
#             DebugLayer(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
#             DebugLayer(),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
#             DebugLayer(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
#             DebugLayer(),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
#             DebugLayer(),
            nn.Dropout(0.1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
#             DebugLayer(),
            nn.Linear(in_features=128 * 3 * 3, out_features=3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3072, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=10),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2),
        )
