import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_train_dataset(name):
    train_set, val_set = get_dataset(name, True)
    return train_set, val_set


def get_test_dataset(name):
    dataset, _ = get_dataset(name, False)
    return dataset


def get_dataset(name, train):
    if name == "mnist":
        return get_mnist(train)
    if name == "svhn":
        return get_svhn(train)
    

def get_mnist(train):
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform_mnist_train)
    if train:
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        return train_set, val_set
    else:
        return dataset, None
    
    
def get_svhn(train):
    transform_svhn_train = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
    ])

    if train:
        split = "train"
        dataset = torchvision.datasets.SVHN(root='./data', split=split, download=True, transform=transform_svhn_train)
        train_set, val_set = torch.utils.data.random_split(dataset, [63257, 10000])
        return train_set, val_set
    else:
        split = "test"
        dataset = torchvision.datasets.SVHN(root='./data', split=split, download=True, transform=transform_svhn_train)
        return dataset, None
    
