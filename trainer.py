from dataset import get_train_dataset, get_test_dataset
from model.component import GRL
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class DANN(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier
        self.discriminator = model.discriminator
        
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        
        self.train_set_src, self.val_set_src = get_train_dataset(params["src"])
        self.train_set_tgt, _ = get_train_dataset(params["tgt"])
        self.test_set_tgt = get_test_dataset(params["tgt"])
        
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.epoch = params["epoch"]

    def training_step(self, batch, batch_idx):
        (inputs_src, targets_src), (inputs_tgt, _) = batch
        device = inputs_src.device
        iterations = self.global_step
        p = float(iterations / self.epoch * (len(self.train_set_src) / self.batch_size))
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        features_src = self.feature_extractor(inputs_src)
        outputs_src = self.classifier(features_src)
        features_src = GRL.apply(features_src, alpha)
        outputs_domain_src = self.discriminator(features_src)
        
        features_tgt = self.feature_extractor(inputs_tgt)
        features_tgt = GRL.apply(features_tgt, alpha)
        outputs_domain_tgt = self.discriminator(features_tgt)
        
        outputs_domain = torch.cat([
            outputs_domain_src,
            outputs_domain_tgt
        ], axis=0)
        targets_domain = torch.cat([
            torch.ones(outputs_domain_src.shape[0]),
            torch.zeros(outputs_domain_tgt.shape[0]),
        ], axis=0).unsqueeze(1).to(device)
        loss_cls = F.cross_entropy(outputs_src, targets_src)
        loss_dsc = F.binary_cross_entropy(outputs_domain, targets_domain)
        loss = loss_cls + loss_dsc
        
        self.log("train_loss_cls", loss_cls, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy(outputs_src, targets_src), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_dsc", loss_dsc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        outputs = self.classifier(features)
        
        loss_cls = F.cross_entropy(outputs, targets)
        self.log("val_acc", self.val_accuracy(outputs, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_cls", loss_cls, on_epoch=True, prog_bar=True, logger=True)
        
        return loss_cls

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        outputs = self.classifier(features)
        
        loss_cls = F.cross_entropy(outputs, targets)
        self.log("test_acc", self.test_accuracy(outputs, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss_cls", loss_cls, on_epoch=True, prog_bar=True, logger=True)
        
        return loss_cls
    
    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()) + list(self.discriminator.parameters()),
            lr=self.lr,
        )

        return optimizer

    def train_dataloader(self):
        src_loader = DataLoader(self.train_set_src, batch_size=self.batch_size, shuffle=True, num_workers=8)
        tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return list(zip(src_loader, tgt_loader))
    
    def val_dataloader(self):
        return DataLoader(self.val_set_src, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_set_tgt, batch_size=self.batch_size, num_workers=8)