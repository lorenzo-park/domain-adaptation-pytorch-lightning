from itertools import cycle
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from backbone.util import get_backbone
from dann.grl import GRL
from dataset.util import get_train_dataset, get_test_dataset


class DANN(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    model = get_backbone(cfg.training.backbone)

    self.feature_extractor = model.feature_extractor
    self.classifier = model.classifier
    self.discriminator = model.discriminator

    self.init_dataset(cfg.dataset.src, cfg.dataset.tgt, cfg.dataset.img_size,
                      cfg.dataset.root, cfg.training.use_tgt_val)

    self.register_buffer("targets_d_src", torch.ones(cfg.training.batch_size).long())
    self.register_buffer("targets_d_tgt", torch.zeros(cfg.training.batch_size).long())

    self.train_accuracy = pl.metrics.Accuracy()
    self.val_accuracy = pl.metrics.Accuracy()
    self.test_accuracy = pl.metrics.Accuracy()

  def training_step(self, batch, batch_idx):
    if self.cfg.training.iterations:
      idx, (inputs_src, targets_src), (inputs_tgt, _) = batch
    else:
      (inputs_src, targets_src), (inputs_tgt, _) = batch

    p = self.get_p()
    lambda_p = self.get_lambda_p(p)

    if self.cfg.training.lr_schedule:
      self.lr_schedule_step(p)

    features_src = self.feature_extractor(inputs_src)
    features_src_rev = GRL.apply(features_src, lambda_p)
    outputs_src = self.classifier(features_src)
    outputs_d_src = self.discriminator(features_src_rev)

    loss_cls = F.cross_entropy(outputs_src, targets_src)
    loss_dsc_src = F.cross_entropy(outputs_d_src, self.targets_d_src)

    features_tgt = self.feature_extractor(inputs_tgt)
    features_tgt_rev = GRL.apply(features_tgt, lambda_p)
    outputs_d_tgt = self.discriminator(features_tgt_rev)

    loss_dsc_tgt = F.cross_entropy(outputs_d_tgt, self.targets_d_tgt)
    loss = loss_cls + loss_dsc_src + loss_dsc_tgt
    train_acc = self.train_accuracy(outputs_src, targets_src)

    # Record logs
    self.log("train_loss_cls", loss_cls, on_step=False, on_epoch=True,
             sync_dist=True)
    self.log("train_loss_dsc", loss_dsc_src + loss_dsc_tgt, on_step=False,
             on_epoch=True, sync_dist=True)
    self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.log("p", p, on_step=False, on_epoch=True, sync_dist=True)
    self.log("train_acc", train_acc, on_step=False, on_epoch=True,
             sync_dist=True)

    return loss

  def training_epoch_end(self, outs):
    self.log("train_acc_epoch", self.train_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    features = self.feature_extractor(inputs)
    outputs = self.classifier(features)

    loss = F.cross_entropy(outputs, targets)
    val_acc = self.val_accuracy(outputs, targets)

    self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
    self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    return loss

  def validation_epoch_end(self, outs):
    self.log("val_acc_epoch", self.val_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def test_step(self, batch, batch_idx):
    inputs, targets = batch
    features = self.feature_extractor(inputs)
    outputs = self.classifier(features)

    loss = F.cross_entropy(outputs, targets)
    test_acc = self.test_accuracy(outputs, targets)

    self.log("test_acc", test_acc, on_step=False, on_epoch=True, logger=True,
             sync_dist=True)
    self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True,
             sync_dist=True)

    return loss

  def test_epoch_end(self, outs):
    test_acc = self.test_accuracy.compute()
    self.log("test_acc_epoch", test_acc, logger=True, sync_dist=True)

  def configure_optimizers(self):
    model_parameter = [
        {
            "params": self.feature_extractor.parameters(),
            "lr_mult": 0.1 if self.cfg.training.backbone == "resnet" else 1.0,
            'decay_mult': 2,
        },
        {
            "params": self.classifier.parameters(),
            "lr_mult": 1.0,
            'decay_mult': 2,
        },
        {
            "params": self.discriminator.parameters(),
            "lr_mult":  1.0,
            'decay_mult': 2,
        }
    ]
    if self.cfg.training.optimizer == "sgd":
      optimizer = torch.optim.SGD(
          model_parameter,
          lr=self.cfg.training.lr,
          momentum=0.9,
          weight_decay=self.cfg.training.weight_decay,
          nesterov=True
      )
    else:
      optimizer = torch.optim.Adam(
          model_parameter,
          lr=self.cfg.training.lr,
          betas=(0.9, 0.999),
          weight_decay=self.cfg.training.weight_decay,
      )


    return optimizer

  def train_dataloader(self):
    src_loader = DataLoader(self.train_set_src, batch_size=self.cfg.training.batch_size,
                            shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True,
                            sampler=None, drop_last=True)
    tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.cfg.training.batch_size,
                            shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True,
                            sampler=None, drop_last=True)
    if self.cfg.training.iterations:
      self.len_dataloader = self.cfg.training.iterations
      return list(zip(range(self.cfg.training.iterations), cycle(src_loader),
                      cycle(tgt_loader)))
    else:
      self.len_dataloader = min(len(src_loader), len(tgt_loader))
      return list(zip(src_loader, tgt_loader))

  def val_dataloader(self):
    return DataLoader(self.val_set_src, batch_size=self.cfg.training.batch_size,
                      num_workers=self.cfg.training.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_set_tgt, batch_size=self.cfg.training.batch_size,
                      num_workers=self.cfg.training.num_workers)

  def init_dataset(self, src, tgt, img_size, root, use_tgt_val):
    self.train_set_src, self.val_set_src = get_train_dataset(src, img_size, root)
    self.train_set_tgt, self.val_set_tgt = get_train_dataset(tgt, img_size, root)
    self.test_set_tgt = get_test_dataset(tgt, img_size, root)

    if use_tgt_val:
      print("####### WARNING #######")
      print("Using target validation set is not valid unsupervised DA setting.")
      print("#######################")
      self.val_set_src = self.test_set_tgt

  def lr_schedule_step(self, p):
    for param_group in self.optimizers().param_groups:
      param_group["lr"] = \
          param_group["lr_mult"] * self.cfg.training.lr / (1 + self.cfg.training.alpha * p) ** self.cfg.training.beta
      param_group["weight_decay"] = \
          self.cfg.training.weight_decay * param_group["decay_mult"]

  def get_p(self):
    current_iterations = self.global_step
    current_epoch = self.current_epoch
    len_dataloader = self.len_dataloader
    p = float(current_iterations / (self.cfg.epoch * len_dataloader))

    return p

  def get_lambda_p(self, p):
    lambda_p = 2. / (1. + np.exp(-self.cfg.training.gamma * p)) - 1

    return lambda_p
