from model.lenet import LeNet
from model.svhn_cnn import SVHNCNN
from pl_model.dann import DANN
from pl_model.source_only import SO


def get_img_size_from_source_dataset(src):
    if src == "mnist":
        return 28
    
    # Default img_size is 32
    return 32


def get_baseline_model(src):
    if src == "mnist":
        return LeNet()
    
    if src == "svhn":
        return SVHNCNN()
    
    return None


def get_da_model(model, base_model, params):
    if model == "dann":
        return DANN(
            model=base_model, 
            params=params
        )
    
    if model == "so" or model == "source_only":
        return SO(
            model=base_model, 
            params=params
        )
    
    return None