from torch.utils.data import DataLoader

import copy
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from backbone.util import get_backbone
from dataset.util import get_train_dataset, get_test_dataset


class ADDA(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.backbone = cfg.training.backbone
    path = os.path.join(cfg.save_path, f"{cfg.dataset.src}2{cfg.dataset.tgt}")
    model = get_backbone(self.backbone, cfg.load, path)

    self.feature_extractor_src = model.feature_extractor
    self.feature_extractor_tgt = copy.deepcopy(model.feature_extractor)

    self.classifier = model.classifier
    self.discriminator = model.discriminator

    self.init_dataset(cfg.dataset.src, cfg.dataset.tgt, cfg.dataset.img_size,
                      cfg.dataset.root, cfg.training.use_tgt_val)

    self.register_buffer("targets_real", torch.ones(
        cfg.training.batch_size).long())
    self.register_buffer("targets_fake", torch.zeros(
        cfg.training.batch_size).long())

    self.disc_accuracy = pl.metrics.Accuracy()
    self.val_accuracy = pl.metrics.Accuracy()
    self.test_accuracy = pl.metrics.Accuracy()

  def training_step(self, batch, batch_idx, optimizer_idx):
    if self.cfg.training.iterations:
      batch_idx, (inputs_src, targets_src), (inputs_tgt, _) = batch
    else:
      (inputs_src, targets_src), (inputs_tgt, _) = batch

    if optimizer_idx == 0:
      inputs_tgt_features = self.feature_extractor_tgt(inputs_tgt)
      outputs_d_tgt = self.discriminator(inputs_tgt_features)
      loss = F.cross_entropy(outputs_d_tgt, self.targets_real)
      self.log("loss_g", loss, on_step=False, on_epoch=True, sync_dist=True)
    else:
      inputs_src_features = self.feature_extractor_src(inputs_src)
      inputs_tgt_features = self.feature_extractor_tgt(inputs_tgt)

      outputs_d_src = self.discriminator(inputs_src_features)
      outputs_d_tgt = self.discriminator(inputs_tgt_features)

      loss = F.cross_entropy(outputs_d_src, self.targets_real) + \
          F.cross_entropy(outputs_d_tgt, self.targets_fake)

      self.last_batch_acc = self.disc_accuracy(
          torch.cat([outputs_d_src, outputs_d_tgt], dim=0),
          torch.cat([self.targets_real, self.targets_fake], dim=0)
      )

      self.log("loss_d", loss, on_step=False, on_epoch=True, sync_dist=True)
      self.log("acc_d", self.last_batch_acc, on_step=False, on_epoch=True,
               sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch

    features = self.feature_extractor_tgt(inputs)
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

    features = self.feature_extractor_tgt(inputs)
    outputs = self.classifier(features)

    loss = F.cross_entropy(outputs, targets)
    test_acc = self.test_accuracy(outputs, targets)

    self.log("test_acc", test_acc, on_step=False,
             on_epoch=True, sync_dist=True)
    self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    return loss

  def test_epoch_end(self, outs):
    self.log("test_acc_epoch", self.test_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def configure_optimizers(self):
    return [
        torch.optim.Adam(
            self.feature_extractor_tgt.parameters(),
            lr=self.cfg.training.lr,
            betas=(0.9, 0.999),
            weight_decay=self.cfg.training.weight_decay,
        ),
        torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.training.lr,
            betas=(0.9, 0.999),
            weight_decay=self.cfg.training.weight_decay,
        )
    ]

  def train_dataloader(self):
    src_loader = DataLoader(self.train_set_src, batch_size=self.cfg.training.batch_size, shuffle=True,
                            num_workers=self.cfg.training.num_workers, pin_memory=True, sampler=None, drop_last=True)
    tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.cfg.training.batch_size, shuffle=True,
                            num_workers=self.cfg.training.num_workers, pin_memory=True, sampler=None, drop_last=True)
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
    self.train_set_src, self.val_set_src = get_train_dataset(
        src, img_size, root)
    self.train_set_tgt, self.val_set_tgt = get_train_dataset(
        tgt, img_size, root)
    self.test_set_tgt = get_test_dataset(tgt, img_size, root)

    if use_tgt_val:
      print("####### WARNING #######")
      print("Using target validation set is not valid unsupervised DA setting.")
      print("#######################")
      self.val_set_src = self.test_set_tgt
