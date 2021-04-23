from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset.util import get_train_dataset, get_test_dataset
from backbone.util import get_backbone


class SO(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    model = get_backbone(cfg.training.backbone)

    self.feature_extractor = model.feature_extractor
    self.classifier = model.classifier

    self.init_dataset(cfg.dataset.src, cfg.dataset.tgt, cfg.dataset.img_size,
                      cfg.dataset.root, cfg.training.use_tgt_val)

    self.train_accuracy = pl.metrics.Accuracy()
    self.val_accuracy = pl.metrics.Accuracy()
    self.test_accuracy = pl.metrics.Accuracy()

  def training_step(self, batch, batch_idx):
    inputs, targets = batch

    features = self.feature_extractor(inputs)
    outputs = self.classifier(features)

    loss = F.cross_entropy(outputs, targets)
    train_acc = self.train_accuracy(outputs, targets)

    self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
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
    model_params = [
        {"params": self.feature_extractor.parameters()},
        {"params": self.classifier.parameters()}
    ]
    if self.cfg.training.optimizer == "adam":
      optimizer = torch.optim.Adam(
          model_params,
          lr=self.cfg.training.lr,
          betas=(0.9, 0.999),
          weight_decay=self.cfg.training.weight_decay,
      )
    else:
      optimizer = torch.optim.SGD(
          model_params,
          lr=self.cfg.training.lr,
          momentum=0.9,
          weight_decay=self.cfg.training.weight_decay,
          nesterov=True
      )

    return optimizer

  def train_dataloader(self):
    return DataLoader(self.train_set_src, batch_size=self.cfg.training.batch_size,
                      shuffle=True, num_workers=self.cfg.training.num_workers)

  def val_dataloader(self):
    return DataLoader(self.val_set_src, batch_size=self.cfg.training.batch_size,
                      num_workers=self.cfg.training.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_set_tgt, batch_size=self.cfg.training.batch_size,
                      num_workers=self.cfg.training.num_workers)

  def init_dataset(self, src, tgt, img_size, root, use_tgt_val):
    self.train_set_src, self.val_set_src = get_train_dataset(src, img_size, root)
    self.test_set_tgt = get_test_dataset(tgt, img_size, root)

    if use_tgt_val:
      print("####### WARNING #######")
      print("Using target validation set is not valid unsupervised DA setting.")
      print("#######################")
      self.val_set_src = self.test_set_tgt
