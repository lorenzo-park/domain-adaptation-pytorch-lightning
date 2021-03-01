from util import get_img_size_from_source_dataset, get_baseline_model, get_da_model
from pytorch_lightning.loggers.neptune import NeptuneLogger

import os
import argparse
import pytorch_lightning as pl


parser = argparse.ArgumentParser(description="Paper experiment results reproduction")
parser.add_argument("--src", type=str, required=True, help="source dataset, e.g. mnist")
parser.add_argument("--tgt", type=str, required=True, help="target dataset, e.g. mnist_m")
parser.add_argument("--disjoint", type=str, required=False, default=None, help="disjoint training dataset, e.g. mnist_m")
parser.add_argument("--model", type=str, required=True, help="domain adaptation model name")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--epoch", type=int, default=20, help="epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr_schedule", type=bool, default=False, help="schedule lr")
parser.add_argument("--use_bottleneck", type=bool, default=False, help="use bottleneck layer")
parser.add_argument("--use_tgt_val", type=bool, default=False, help="Use target validation")
parser.add_argument("--fe_lr", type=float, default=0.1, help="Use target validation")
parser.add_argument("--cls_lr", type=float, default=0.1, help="Use target validation")
parser.add_argument("--disc_lr", type=float, default=1.0, help="Use target validation")

args = parser.parse_args()

params = {
    "model": args.model,
    "batch_size": args.batch_size,
    "epoch": args.epoch,
    "lr": args.lr,
    "src": args.src,
    "tgt": args.tgt,
    "disjoint": args.disjoint,
    "use_tgt_val": args.use_tgt_val,
    "img_size": get_img_size_from_source_dataset(args.src),
    "lr_schedule": args.lr_schedule,
    "use_bottleneck": args.use_bottleneck,
    "gamma": 10,
    "alpha": 10,
    "beta": 0.75,
    "seed": 42,
    "fe_lr": args.fe_lr,
    "cls_lr": args.cls_lr,
    "disc_lr": args.disc_lr,
}

print(params)

pl.seed_everything(params["seed"])

base_model = get_baseline_model(args.src, args.use_bottleneck)
assert base_model is not None

model = get_da_model(args.model, base_model, params)
assert model is not None

logger = NeptuneLogger(
    api_key=os.environ.get("NETPUNE_API_TOKEN"),
    project_name="kaggle.lorenzo.park/domain-adaptation",
    params=params,
    experiment_name=f"{params['src']}2{params['tgt']}",
    tags=[f"{model.__class__.__name__}", f"{base_model.__class__.__name__}", "lorenzo-lab"]
)

# logger = pl.loggers.TestTubeLogger("output", name=)
# logger.log_hyperparams(params)

trainer = pl.Trainer(
    deterministic=True,
    check_val_every_n_epoch=1, 
    gpus=1,
    logger=logger,
    max_epochs=params["epoch"],
    weights_summary="top",
)
trainer.fit(model)
trainer.test()
