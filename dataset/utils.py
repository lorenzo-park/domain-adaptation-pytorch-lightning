from torch.utils.data import DataLoader, random_split
from dataset.mnist_m import MNISTM
from dataset.office31 import Office31
from dataset.officehome import OfficeHome

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
    if name in ["A", "W", "D"]:
        return get_office31(train, img_size, name)
    if name in ["Ar", "Cl", "Pr", "Rw"]:
        return get_officehome(train, img_size, name)

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
    
    
def get_office31(train, img_size, domain_type):
    #     transform_train = transforms.Compose([
    #         transforms.Resize(227),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    transform_train = transforms.Compose([
        transforms.Lambda(lambda image: image.resize((256, 256))),
        transforms.RandomResizedCrop(size=(img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    start_first = 0
    start_center = (256 - img_size - 1) / 2
    start_last = 256 - img_size - 1

    transform_test = transforms.Compose([
        transforms.Lambda(lambda image: image.resize((256, 256))),
        PlaceCrop(img_size, start_center, start_center),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset_train = Office31(root="./data/office31", task=domain_type, download=True, transform=transform_train)
    dataset_test = Office31(root="./data/office31", task=domain_type, download=True, transform=transform_test)

    if train:
        return dataset_train, dataset_test
    else:
        return dataset_test, None


def get_officehome(train, img_size, domain_type):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda image: image.resize((256, 256))),
        transforms.RandomResizedCrop(size=(img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = OfficeHome(root="./data/officehome", task=domain_type, download=True, transform=transform_train)

    if train:
        return dataset, dataset
    else:
        return dataset, None

    
class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))