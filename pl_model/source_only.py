from dataset.utils import get_train_dataset, get_test_dataset
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class SO(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier
        
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        
        self.train_set_src, self.val_set_src = get_train_dataset(params["src"], params["img_size"])
        self.test_set_tgt = get_test_dataset(params["tgt"], params["img_size"])
        
        if params["use_tgt_val"]:
            print("####### WARNING #######")
            print("Using target validation set is not valid unsupervised d.a. setting.")
            print("#######################")
            self.val_set_src = self.test_set_tgt
        
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.epochs = params["epoch"]
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        
        self.lr_schedule = params["lr_schedule"]

    def training_step(self, batch, batch_idx):
        inputs_src, targets_src = batch
        device = inputs_src.device
        
        iterations = self.global_step
        p = float(iterations / (self.epochs * (len(self.train_set_src) // self.batch_size)))
        
        if self.lr_schedule:
            # Schedule learning rate
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = self.lr / (1. + self.alpha * p) ** self.beta
        
        features_src = self.feature_extractor(inputs_src)
        outputs_src = self.classifier(features_src)
        
        loss = F.cross_entropy(outputs_src, targets_src)
        
        self.log("train_loss", loss, on_epoch=True, logger=True)
        self.log("train_acc", self.train_accuracy(outputs_src, targets_src), on_epoch=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        outputs = self.classifier(features)
        
        loss_cls = F.cross_entropy(outputs, targets)
        self.log("val_acc", self.val_accuracy(outputs, targets), on_epoch=True, logger=True)
        self.log("val_loss_cls", loss_cls, on_epoch=True, logger=True)
        
        return loss_cls

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        outputs = self.classifier(features)
        
        loss_cls = F.cross_entropy(outputs, targets)
        self.log("test_acc", self.test_accuracy(outputs, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss_cls", loss_cls, on_epoch=True, logger=True)
        
        return loss_cls
    
    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
            momentum=0.9,
        )

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set_src, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_set_src, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_set_tgt, batch_size=self.batch_size, num_workers=8)