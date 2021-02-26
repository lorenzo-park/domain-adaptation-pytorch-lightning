from torch.utils.data import DataLoader, random_split
from dataset.mnist_m import MNISTM

import torch
import torchvision
import torchvision.transforms as transforms


def get_src_tgt_datasets(src, tgt, img_size=32):
    train_set, val_set = get_train_dataset(name, img_size)
    test_set = get_test_dataset(name, img_size)

    return train_set, val_set, test_set


def get_train_dataset(name, img_size=32):
    train_set, val_set = get_dataset(name, True, img_size)
    return train_set, val_set


def get_test_dataset(name, img_size=32):
    dataset, _ = get_dataset(name, False, img_size)
    return dataset


def get_dataset(name, train, img_size):
    if name == "mnist":
        return get_mnist(train, img_size)
    if name == "emnist_letters":
        return get_emnist_letters(train, img_size)
    if name == "svhn":
        return get_svhn(train, img_size)
    if name == "mnist_m":
        return get_mnist_m(train, img_size)

def get_mnist(train, img_size):
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform_mnist_train)
    if train:
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        return train_set, val_set
    else:
        return dataset, None


def get_emnist_letters(train, img_size):
    transform_mnist_train = transforms.Compose([
#         transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    
    dataset = torchvision.datasets.EMNIST(root='./data', train=train, split="letters", download=True, transform=transform_mnist_train)
    dataset_len = len(dataset)
    if train:
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, dataset_len-50000])
        val_set, _ = torch.utils.data.random_split(val_set, [10000, len(val_set)-10000])
        return train_set, None
    else:
        return dataset, None
    

def get_svhn(train, img_size):
    transform_svhn_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
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
    
    
def get_mnist_m(train, img_size):
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    dataset = MNISTM(root='./data', train=train, download=True, transform=transform_mnist_train)
    if train:
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        return train_set, val_set
    else:
        return dataset, None