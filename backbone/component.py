import torch.nn as nn


class DebugLayer(nn.Module):
  def forward(self, x):
    print(x.shape)
    return x


class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)