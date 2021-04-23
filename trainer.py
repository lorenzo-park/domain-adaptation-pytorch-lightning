import argparse
import hydra
import os
import torch
import pytorch_lightning as pl

from util import get_da_model


def save_backbone(model, path):
  if not os.path.exists(path):
    os.makedirs(path)
  feature_extractor_path = os.path.join(path, "feature_extractor.pth")
  classifier_path = os.path.join(path, "classifier.pth")
  torch.save(model.feature_extractor.state_dict(), feature_extractor_path)
  torch.save(model.classifier.state_dict(), classifier_path)

@hydra.main(config_name="config")
def run(cfg):
    print(cfg)
    pl.seed_everything(cfg.seed)
    if cfg.logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project="dcuda",
            name=f"{cfg.dataset.src}2{cfg.dataset.tgt}"
        )
    else:
        logger = pl.loggers.TestTubeLogger(
            "output", name=f"{cfg.dataset.src}2{cfg.dataset.tgt}")
        logger.log_hyperparams(cfg)

    model = get_da_model(cfg)
    trainer = pl.Trainer(
        deterministic=True,
        check_val_every_n_epoch=1,
        gpus=cfg.gpus,
        logger=logger,
        max_epochs=cfg.epoch,
        weights_summary="top",
        accelerator='ddp',
    )

    trainer.fit(model)
    trainer.test()

    if cfg.save:
        save_backbone(model, cfg.save)

if __name__ == '__main__':
  run()
