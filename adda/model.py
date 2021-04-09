from torch.utils.data import DataLoader

import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from backbone.util import get_backbone
from dataset.util import get_train_dataset, get_test_dataset


class ADDA(pl.LightningModule):
  def __init__(self, params):
    super().__init__()
    self.backbone = params["backbone"]
    model = get_backbone(self.backbone, params["load"])
    
    self.feature_extractor_src = model.feature_extractor
    self.feature_extractor_tgt = copy.deepcopy(model.feature_extractor)
    
    self.classifier = model.classifier
    self.discriminator = model.discriminator
    
    self.init_dataset(params["src"], params["tgt"], params["img_size"], 
                      params["irt"], params["use_tgt_val"])
    
    self.batch_size = params["batch_size"]
    self.lr = params["lr"]
    self.momentum = params["momentum"]
    self.optimizer = params["optimizer"]
    self.weight_decay = params["weight_decay"]
    self.iterations = params["iterations"]
    
    self.register_buffer("targets_real", torch.ones(self.batch_size).long())
    self.register_buffer("targets_fake", torch.zeros(self.batch_size).long())
    
    self.disc_accuracy = pl.metrics.Accuracy()
    self.val_accuracy = pl.metrics.Accuracy()
    self.test_accuracy = pl.metrics.Accuracy()
    
  def training_step(self, batch, batch_idx, optimizer_idx):
    if self.iterations:
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
    
    self.log("test_acc", test_acc, on_step=False, on_epoch=True, sync_dist=True)
    self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
      
    return loss
  
  def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), 
                 prog_bar=True, logger=True, sync_dist=True)
        
  def configure_optimizers(self):
    return [
      torch.optim.Adam(
        self.feature_extractor_tgt.parameters(),
        lr=self.lr,
        betas=(0.9, 0.999),
        weight_decay=self.weight_decay,
      ),
      torch.optim.Adam(
        self.discriminator.parameters(),
        lr=self.lr,
        betas=(0.9, 0.999),
        weight_decay=self.weight_decay,
      )
  ]
    
  def train_dataloader(self):
    src_loader = DataLoader(self.train_set_src, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)
    tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)
    if self.iterations:
        self.len_dataloader = self.iterations
        return list(zip(range(self.iterations), cycle(src_loader), cycle(tgt_loader)))
    else:
        self.len_dataloader = min(len(src_loader), len(tgt_loader))
        return list(zip(src_loader, tgt_loader))
  
  def val_dataloader(self):
    return DataLoader(self.val_set_src, batch_size=self.batch_size, num_workers=8)
  
  def test_dataloader(self):
    return DataLoader(self.test_set_tgt, batch_size=self.batch_size, num_workers=8)
        
    
  def init_dataset(self, src, tgt, img_size, irt, use_tgt_val):
    self.train_set_src, self.val_set_src = get_train_dataset(src, img_size)
    self.train_set_tgt, self.val_set_tgt = get_train_dataset(tgt, img_size)
    self.test_set_tgt = get_test_dataset(tgt, img_size)
    
    if irt:
      self.train_set_tgt, _ = get_train_dataset(irt, img_size)

    if use_tgt_val:
      print("####### WARNING #######")
      print("Using target validation set is not valid unsupervised DA setting.")
      print("#######################")
      self.val_set_src = self.test_set_tgt