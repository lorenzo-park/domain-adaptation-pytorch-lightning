from so.model import SO
from dann.model import DANN
from adda.model import ADDA


def get_da_model(cfg):
  if cfg.model == "so":
    return SO(cfg)

  if cfg.model == "dann":
    return DANN(cfg)

  if cfg.model == "adda":
    return ADDA(cfg)
