import argparse
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Experiment script")
  # Experiment params
  parser.add_argument(
      "--src",
      type=str, required=True, help="source dataset, e.g. mnist"
  )
  parser.add_argument(
      "--tgt",
      type=str, required=True, help="target dataset, e.g. mnist-m"
  )
  parser.add_argument(
      "--backbone",
      type=str, required=True, help="backbone feature extractor and classifier"
  )
  parser.add_argument(
      "--model",
      type=str, required=True, help="domain adaptation model, e.g. dann"
  )
  parser.add_argument(
      "--irt",
      type=str, required=False, default=None, help="irrelevant task dataset"
  )
  parser.add_argument(
      "--img_size",
      type=int, required=True, help="image input size"
  )

  # Training params
  parser.add_argument(
      "--lr",
      type=float, required=True,
      help="learning rate"
  )
  parser.add_argument(
      "--weight_decay",
      type=float, required=True, help="batch size"
  )
  parser.add_argument(
      "--momentum",
      type=float, required=True, help="momentum"
  )
  parser.add_argument(
      "--optimizer",
      type=str, required=True, help="optimizer type"
  )
  parser.add_argument(
      "--batch_size",
      type=int, required=True, help="batch size"
  )
  parser.add_argument(
      "--epoch",
      type=int, required=True, help="epochs"
  )
  parser.add_argument(
      "--iterations",
      type=int, required=False, default=None,
      help="if dataset size of src and tgt are different, use this option."
  )
  parser.add_argument(
      "--lr_schedule",
      type=bool, required=False, default=False,
      help="learning rate scheduling"
  )

  # Other params
  parser.add_argument(
      "--logger",
      type=bool, required=True
  )
  parser.add_argument(
      "--save",
      type=str, required=False, default=None,
      help="path to save model weights"
  )
  parser.add_argument(
      "--load",
      type=str, required=False, default=None,
      help="path to load model weights"
  )
  parser.add_argument(
      "--seed",
      type=int, required=False, default=8888,
      help="random seed"
  )
  parser.add_argument(
      "--use_tgt_val",
      type=bool, required=False, default=False,
      help="target domain as validation set"
  )
  parser.add_argument(
      "--gpus",
      type=int, required=False, default=1,
      help="ddp gpus"
  )

  params = vars(parser.parse_args())

  pl.seed_everything(params["seed"])
  if params["logger"]:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(
        project="dcuda",
        name=f"{params['src']}2{params['tgt']}"
    )
  else:
    logger = pl.loggers.TestTubeLogger(
        "output", name=f"{params['src']}2{params['tgt']}")
    logger.log_hyperparams(params)

  model = get_da_model(params)
  trainer = pl.Trainer(
      deterministic=True,
      check_val_every_n_epoch=1,
      gpus=params["gpus"],
      logger=logger,
      max_epochs=params["epoch"],
      weights_summary="top",
      accelerator='ddp',
  )

  trainer.fit(model)
  trainer.test()

  if params["save"]:
    save_backbone(model, params["save"])
