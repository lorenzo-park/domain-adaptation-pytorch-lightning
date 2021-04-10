import torch.nn as nn

from backbone.component import Flatten, DebugLayer


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
        nn.Linear(in_features=768, out_features=100),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=100, out_features=100),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=100, out_features=10),
    )
    self.discriminator = nn.Sequential(
        nn.Linear(in_features=768, out_features=500),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=500, out_features=500),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=500, out_features=2),
    )
