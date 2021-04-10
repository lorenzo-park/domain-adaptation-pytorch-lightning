import torch.nn as nn

from backbone.component import Flatten, DebugLayer


class SVHNDANN(nn.Module):
  def __init__(self):
    super(SVHNDANN, self).__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        Flatten(),
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features=128 * 4 * 4, out_features=3072),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=3072, out_features=2048),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=2048, out_features=10),
    )

    self.discriminator = nn.Sequential(
        nn.Linear(in_features=128 * 4 * 4, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=1024, out_features=2),
    )


class SVHNADDA(nn.Module):
  def __init__(self):
    super(SVHNADDA, self).__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.3),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.Dropout2d(0.5),
        nn.ReLU(),
        Flatten(),
        nn.Linear(256*4*4, 500),
        nn.ReLU(),
    )

    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(500, 10),
    )

    self.discriminator = nn.Sequential(
        nn.Linear(in_features=500, out_features=500),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=500, out_features=500),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=500, out_features=2),
    )
