from dataset.utils import get_train_dataset, get_test_dataset
from model.component import GRL
from model.resnet import ResNet
from torch.utils.data import DataLoader

import torch
import torchsummary
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class DANN(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        if model.__class__ == ResNet:
            self.finetune = True
            
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier
        self.discriminator = model.discriminator

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        
        self.train_set_src, self.val_set_src = get_train_dataset(params["src"], params["img_size"])
        self.train_set_tgt, self.val_set_tgt = get_train_dataset(params["tgt"], params["img_size"])
        
        if params["disjoint"]:
            self.train_set_tgt, _ = get_train_dataset(params["disjoint"], params["img_size"])
        
        if params["use_tgt_val"]:
            print("####### WARNING #######")
            print("Using target validation set is not valid unsupervised d.a. setting.")
            print("#######################")
            self.val_set_src = self.val_set_tgt
        
        self.test_set_tgt = get_test_dataset(params["tgt"], params["img_size"])
        
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.epochs = params["epoch"]
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        
        self.lr_schedule = params["lr_schedule"]
        self.use_bottleneck = params["use_bottleneck"]
        
        if self.finetune:
            self.model_parameter = [
                {
                    'params': self.feature_extractor.parameters(), 
                    "lr": self.lr * 0.1,
                },
                {
                    'params': self.classifier.parameters(),
                },
                {
                    'params': self.discriminator.parameters(),
                }
            ]
        else:
            self.model_parameter = [
                {'params': self.feature_extractor.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.discriminator.parameters()}
            ]
        
        if self.use_bottleneck:
            self.bottleneck = model.bottleneck
            self.model_parameter.append({'params': self.bottleneck.parameters()})

    def training_step(self, batch, batch_idx):
        (inputs_src, targets_src), (inputs_tgt, _) = batch
        device = inputs_src.device
        
        # Calculate p
        iterations = self.global_step
        p = float(iterations / (self.epochs * (len(self.train_set_src) // self.batch_size)))
        
        if self.lr_schedule:
            # Schedule learning rate
            if self.finetune:
                for param_group in self.optimizers().param_groups[1:]:
                    param_group["lr"] = self.lr / (1. + self.alpha * p) ** self.beta
            else:
                for param_group in self.optimizers().param_groups:
                    param_group["lr"] = self.lr / (1. + self.alpha * p) ** self.beta
        
        # Calculate classification loss
        features_src = self.feature_extractor(inputs_src)
        if self.use_bottleneck:
            features_src = self.bottleneck(features_src)
        outputs_src = self.classifier(features_src)
        loss_cls = F.nll_loss(F.log_softmax(outputs_src, 1), targets_src)
        
        # Calculate domain discrimination loss
        lambda_p = 2. / (1. + np.exp(-self.gamma * p)) - 1
        features_src = GRL.apply(features_src, lambda_p)
        outputs_domain_src = self.discriminator(features_src)
        
        features_tgt = self.feature_extractor(inputs_tgt)
        if self.use_bottleneck:
            features_tgt = self.bottleneck(features_tgt)
        features_tgt = GRL.apply(features_tgt, lambda_p)
        outputs_domain_tgt = self.discriminator(features_tgt)
        
        targets_domain_src = torch.ones(outputs_domain_src.shape[0]).to(device).long()
        targets_domain_tgt = torch.zeros(outputs_domain_tgt.shape[0]).to(device).long()
        loss_dsc_src = F.nll_loss(F.log_softmax(outputs_domain_src, 1), targets_domain_src)
        loss_dsc_tgt = F.nll_loss(F.log_softmax(outputs_domain_tgt, 1), targets_domain_tgt)
        loss = loss_cls + loss_dsc_src + loss_dsc_tgt
        
        # Record logs
        self.log("train_loss_cls", loss_cls, on_epoch=True)
        self.log("train_acc", self.train_accuracy(outputs_src, targets_src), on_epoch=True)
        self.log("train_loss_dsc", loss_dsc_src + loss_dsc_tgt, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        outputs = self.classifier(features)
        
        loss_cls = F.nll_loss(F.log_softmax(outputs, 1), targets)
        self.log("val_acc", self.val_accuracy(outputs, targets), on_epoch=True)
        self.log("val_loss_cls", loss_cls, on_epoch=True)
        
        return loss_cls

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        outputs = self.classifier(features)
        
        loss_cls = F.nll_loss(F.log_softmax(outputs, 1), targets)
        self.log("test_acc", self.test_accuracy(outputs, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss_cls", loss_cls, on_epoch=True, logger=True)
        
        return loss_cls
    
    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model_parameter,
            lr=self.lr,
            momentum=0.9,
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