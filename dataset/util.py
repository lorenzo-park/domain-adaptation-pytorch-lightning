from torch.utils.data import DataLoader, random_split
from dataset.mnist_m import MNISTM
from dataset.emnist_m_letters import EMNISTMLetters
from dataset.fashionmnist_m import FashionMNISTM
from dataset.office31 import Office31
from dataset.officehome import OfficeHome
from dataset.stylized_dataset import StylizedDataset

import torch
import torchvision
import torchvision.transforms as transforms


def get_src_tgt_datasets(src, tgt, img_size, root):
  train_set, val_set = get_train_dataset(name, img_size, root)
  test_set = get_test_dataset(name, img_size, root)

  return train_set, val_set, test_set


def get_train_dataset(name, img_size, root):
  train_set, val_set = get_dataset(name, True, img_size, root)
  return train_set, val_set


def get_test_dataset(name, img_size, root):
  dataset, _ = get_dataset(name, False, img_size, root)
  return dataset


def get_dataset(name, train, img_size, root):
  if name == "mnist":
    dataset = get_mnist(train, img_size, root)
  if name == "emnist_letters":
    dataset = get_emnist_letters(train, img_size, root)
  if name == "fmnist":
    dataset = get_fmnist(train, img_size, root)
  if name == "svhn":
    dataset = get_svhn(train, img_size, root)
  if name == "mnist_m":
    dataset = get_mnist_m(train, img_size, root)
  if name == "emnist_m_letters":
    dataset = get_emnist_m_letters(train, img_size, root)
  if name == "fmnist_m":
    dataset = get_fmnist_m(train, img_size, root)
  if name in ["A", "W", "D"]:
    dataset = get_office31(train, img_size, name, root)
  if name in ["Ar", "Cl", "Pr", "Rw"]:
    dataset = get_officehome(train, img_size, name, root)
  if name == "svhn2mnist":
    dataset = get_svhn2mnist(train, img_size, root)
  if name == "mnist2mnist_m":
    dataset = get_mnist2mnist_m(train, img_size, root)

  return dataset


def get_svhn2mnist(train, img_size, root):
  if img_size > 32:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  dataset = StylizedDataset(
      root=f"{root}/svhn2mnist", classes=[str(i) for i in range(10)], transform=transform_mnist_train)

  return dataset, None


def get_mnist2mnist_m(train, img_size, root):
  if img_size > 32:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  dataset = StylizedDataset(root=f"{root}/mnist2mnist_m",
                            classes=[str(i) for i in range(10)], transform=transform_mnist_train)

  return dataset, None


def get_mnist(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])

  dataset = torchvision.datasets.MNIST(
      root=root, train=train, download=True, transform=transform_mnist_train)
  if train:
    train_set, val_set = torch.utils.data.random_split(
        dataset, [50000, 10000])
    return train_set, val_set
  else:
    return dataset, None


def get_emnist_letters(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])

  dataset = torchvision.datasets.EMNIST(
      root=root, train=train, split="letters", download=True, transform=transform_mnist_train)
  dataset_len = len(dataset)
  if train:
    train_set, val_set = torch.utils.data.random_split(
        dataset, [50000, dataset_len-50000])
    val_set, _ = torch.utils.data.random_split(
        val_set, [10000, len(val_set)-10000])
    return train_set, None
  else:
    return dataset, None


def get_fmnist(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ChanDup(),
    ])

  dataset = torchvision.datasets.FashionMNIST(
      root=root, train=train, download=True, transform=transform_mnist_train)
  dataset_len = len(dataset)
  if train:
    train_set, val_set = torch.utils.data.random_split(
        dataset, [50000, dataset_len-50000])
    return train_set, None
  else:
    return dataset, None


def get_emnist_m_letters(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  dataset = EMNISTMLetters(root=root, download=True,
                           transform=transform_mnist_train)

  train_set, val_set = torch.utils.data.random_split(dataset, [50000, 12400])
  return train_set, None


def get_fmnist_m(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  dataset = FashionMNISTM(root=root, download=True,
                          transform=transform_mnist_train)

  train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
  return train_set, None


def get_svhn(train, img_size, root):
  if img_size > 32:
    transform_svhn_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_svhn_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  if train:
    split = "train"
    dataset = torchvision.datasets.SVHN(
        root=root, split=split, download=True, transform=transform_svhn_train)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [63257, 10000])
    return train_set, val_set
  else:
    split = "test"
    dataset = torchvision.datasets.SVHN(
        root=root, split=split, download=True, transform=transform_svhn_train)
    return dataset, None


def get_mnist_m(train, img_size, root):
  if img_size > 28:
    transform_mnist_train = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
  else:
    transform_mnist_train = transforms.Compose([
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  dataset = MNISTM(root=root, train=train, download=True,
                   transform=transform_mnist_train)
  if train:
    train_set, val_set = torch.utils.data.random_split(
        dataset, [50000, 10000])
    return train_set, val_set
  else:
    return dataset, None


def get_office31(train, img_size, domain_type, root):
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
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225]),
  ])
  start_first = 0
  start_center = (256 - img_size - 1) / 2
  start_last = 256 - img_size - 1

  transform_test = transforms.Compose([
      transforms.Lambda(lambda image: image.resize((256, 256))),
      PlaceCrop(img_size, start_center, start_center),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225]),
  ])

  dataset_train = Office31(
      root=f"{root}/office31", task=domain_type, download=True, transform=transform_train)
  dataset_test = Office31(
      root=f"{root}/office31", task=domain_type, download=True, transform=transform_test)

  if train:
    return dataset_train, dataset_test
  else:
    return dataset_test, None


def get_officehome(train, img_size, domain_type, root):
  #     transform_train = transforms.Compose([
  #         transforms.Lambda(lambda image: image.resize((256, 256))),
  #         transforms.RandomResizedCrop(size=(img_size, img_size)),
  #         transforms.RandomHorizontalFlip(),
  #         transforms.ToTensor(),
  #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  #     ])
  transform_train = transforms.Compose([
      transforms.Lambda(lambda image: image.resize((256, 256))),
      transforms.RandomResizedCrop(size=(img_size, img_size)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225]),
  ])
  start_first = 0
  start_center = (256 - img_size - 1) / 2
  start_last = 256 - img_size - 1

  transform_test = transforms.Compose([
      transforms.Lambda(lambda image: image.resize((256, 256))),
      PlaceCrop(img_size, start_center, start_center),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225]),
  ])

  dataset_train = OfficeHome(
      root=f"{root}/officehome", task=domain_type, download=True, transform=transform_train)
  dataset_test = OfficeHome(
      root=f"{root}/officehome", task=domain_type, download=True, transform=transform_test)

  if train:
    return dataset_train, dataset_test
  else:
    return dataset_test, None


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


class ChanDup:
  def __call__(self, img):
    return img.repeat(3, 1, 1)
