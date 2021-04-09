from so.model import SO
from dann.model import DANN
from adda.model import ADDA


def get_da_model(params):
  if params["model"] == "so":
    return SO(params)

  if params["model"] == "dann":
    return DANN(params)
  
  if params["model"] == "adda":
    return ADDA(params)