from model.lenet import LeNet
from model.svhn_cnn import SVHNCNN
from model.resnet import resnet50
from pl_model.dann import DANN
from pl_model.source_only import SO


def get_img_size_from_source_dataset(src):
    # Default img_size is 32
    if src == "mnist":
        return 28
    
    if src == "svhn":
        return 28
    
    if src in ["A", "W", "D"]:
        return 224
    
    if src in ["Ar", "Cl", "Pr", "Rw"]:
        return 224


def get_baseline_model(src, use_bottleneck=False):
    if src == "mnist":
        print("#### LENET ####")
        return LeNet()
    
    if src == "svhn":
        print("#### SVHNCNN ####")
        return SVHNCNN()
    
    if src in ["A", "W", "D"]:
        if use_bottleneck:
            print("#### RESNET with bottleneck ####")
        else:
            print("#### RESNET ####")
        return resnet50(pretrained=True, bottleneck=use_bottleneck, classes=31)
    
    if src in ["Ar", "Cl", "Pr", "Rw"]:
        if use_bottleneck:
            print("#### RESNET with bottleneck ####")
        else:
            print("#### RESNET ####")
        return resnet50(pretrained=True, bottleneck=use_bottleneck, classes=65)
    
    
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
