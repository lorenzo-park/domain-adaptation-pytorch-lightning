from model.component import Flatten

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
#             DebugLayer(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), (2,2)),
#             DebugLayer(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
#             DebugLayer(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), (2,2)),
#             DebugLayer(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
#             DebugLayer(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), (2,2)),
#             DebugLayer(),
            Flatten(),
        )

        self.classifier = nn.Sequential(
#             DebugLayer(),
            nn.Linear(in_features=128*3*3, out_features=3072),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3072, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=10),
            nn.Sigmoid(),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid(),
        )