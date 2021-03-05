from dataset.utils import get_train_dataset, get_test_dataset
from model.component import GRL
from model.resnet import ResNet
from torch.utils.data import DataLoader
from itertools import cycle

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
        self.momentum = params["momentum"]
        self.weight_decay = params["weight_decay"]
        
        self.lr_schedule = params["lr_schedule"]
        self.use_bottleneck = params["use_bottleneck"]
        
        if self.finetune:
            self.model_parameter = [
                {
                    "params": self.feature_extractor.parameters(), 
                    "lr_mult": params["fe_lr"],
                    'decay_mult': 2,
                },
                {
                    "params": self.classifier.parameters(),
                    "lr_mult": params["cls_lr"],
                    'decay_mult': 2,
                },
                {
                    "params": self.discriminator.parameters(),
                    "lr_mult": params["disc_lr"],
                    'decay_mult': 2,
                }
            ]
        else:
            self.model_parameter = [
                {"params": self.feature_extractor.parameters()},
                {"params": self.classifier.parameters()},
                {"params": self.discriminator.parameters()}
            ]
        
        if self.use_bottleneck:
            self.bottleneck = model.bottleneck
            self.model_parameter.append({
                "params": self.bottleneck.parameters(),
                "lr_mult": 10,
            })
            
        self.iters_per_epoch = params["iters_per_epoch"]

    def training_step(self, batch, batch_idx):
        if self.iters_per_epoch:
            idx, (inputs_src, targets_src), (inputs_tgt, _) = batch
        else:
            (inputs_src, targets_src), (inputs_tgt, _) = batch
            
        device = inputs_src.device
        
        # Calculate p
        iterations = self.global_step
        current_epoch = self.current_epoch
        len_dataloader = self.len_dataloader
        p = float(iterations / (self.epochs * len_dataloader))
        lambda_p = 2. / (1. + np.exp(-self.gamma * p)) - 1
        
        if self.lr_schedule:
            # Schedule learning rate
            if self.finetune:
                for param_group in self.optimizers().param_groups:
                    param_group["lr"] = param_group["lr_mult"] * self.lr / (1. + self.alpha * p) ** self.beta
                    param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']
#                 print(list(map(lambda x: x["lr"], self.optimizers().param_groups)))
            else:
                for param_group in self.optimizers().param_groups:
                    param_group["lr"] = self.lr / (1. + self.alpha * p) ** self.beta
        
        targets_domain_src = torch.ones(inputs_src.shape[0]).long().to(device)
        targets_domain_tgt = torch.zeros(inputs_tgt.shape[0]).long().to(device)
        
        # Calculate classification loss
        features_src = self.feature_extractor(inputs_src)
        features_src_rev = GRL.apply(features_src, lambda_p)
        outputs_src = self.classifier(features_src)
        outputs_domain_src = self.discriminator(features_src_rev)
        
        loss_cls = F.cross_entropy(outputs_src, targets_src)
        loss_dsc_src = F.cross_entropy(outputs_domain_src, targets_domain_src)
        
        features_tgt = self.feature_extractor(inputs_tgt)
        features_tgt_rev = GRL.apply(features_tgt, lambda_p)
        outputs_domain_tgt = self.discriminator(features_tgt_rev)
        
        loss_dsc_tgt = F.cross_entropy(outputs_domain_tgt, targets_domain_tgt)
        loss = loss_cls + loss_dsc_src + loss_dsc_tgt
        
        # Record logs
        self.log("train_loss_cls", loss_cls, on_epoch=True)
        self.log("train_acc", self.train_accuracy(outputs_src, targets_src), on_epoch=True)
        self.log("train_loss_dsc", loss_dsc_src + loss_dsc_tgt, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        self.log("p", p, on_epoch=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.feature_extractor(inputs)
        outputs = self.classifier(features)
        
        loss_cls = F.cross_entropy(outputs, targets)
        self.log("val_acc", self.val_accuracy(outputs, targets), on_epoch=True)
        self.log("val_loss_cls", loss_cls, on_epoch=True)
        
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
        self.log("test_acc_epoch", self.test_accuracy.compute(), on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model_parameter,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )

        return optimizer

    def train_dataloader(self):
        src_loader = DataLoader(self.train_set_src, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)
        tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)
        if self.iters_per_epoch:
            self.len_dataloader = self.iters_per_epoch
            return list(zip(range(self.iters_per_epoch), cycle(src_loader), cycle(tgt_loader)))
        else:
            self.len_dataloader = min(len(src_loader), len(tgt_loader))
            return list(zip(src_loader, tgt_loader))
    
    def val_dataloader(self):
        return DataLoader(self.val_set_src, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_set_tgt, batch_size=self.batch_size, num_workers=8)