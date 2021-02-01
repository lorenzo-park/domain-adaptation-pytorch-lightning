from util import get_img_size_from_source_dataset, get_baseline_model, get_da_model

import argparse
import pytorch_lightning as pl



parser = argparse.ArgumentParser(description="Paper experiment results reproduction")
parser.add_argument("--src", type=str, required=True, help="source dataset, e.g. mnist")
parser.add_argument("--tgt", type=str, required=True, help="target dataset, e.g. mnist_m")
parser.add_argument("--model", type=str, required=True, help="domain adaptation model name")

args = parser.parse_args()

params = {
    "batch_size": 64,
    "epoch": 20,
    "lr": 1e-2,
    "src": args.src,
    "tgt": args.tgt,
    "img_size": get_img_size_from_source_dataset(args.src),
    "gamma": 10,
    "alpha": 10,
    "beta": 0.75,
    "seed": 42
}

pl.seed_everything(params["seed"])

logger = pl.loggers.TestTubeLogger("output", name=f"{params['src']}2{params['tgt']}")
logger.log_hyperparams(params)

base_model = get_baseline_model(args.src)
assert base_model is not None

model = get_da_model(args.model, base_model, params)
assert model is not None

trainer = pl.Trainer(
    deterministic=True,
    check_val_every_n_epoch=1, 
    gpus=1,
    logger=logger,
    max_epochs=params["epoch"],
)
trainer.fit(model)
trainer.test()
