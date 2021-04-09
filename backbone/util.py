import os
import torch

from backbone.lenet import LeNet
from backbone.svhn import SVHNDANN, SVHNADDA
from backbone.resnet import ResNet

def get_backbone(backbone, load=None):
  if backbone == "lenet":
    model = LeNet()
  
  if backbone == "svhn-dann":
    model = SVHNDANN()
    
  if backbone == "svhn-adda":
    model = SVHNADDA()
    
  if backbone == "resnet50-32":
    model = ResNet(32, bottleneck=False, pretrained=True)
    
  if backbone == "resnet50-65":
    model = ResNet(65, bottleneck=False, pretrained=True)
    
  if load:
    feature_extractor_path = os.path.join(load, "feature_extractor.pth")
    model.feature_extractor.load_state_dict(torch.load(feature_extractor_path))

    classifier_path = os.path.join(load, "classifier.pth")
    model.classifier.load_state_dict(torch.load(classifier_path))
    
      
  return model