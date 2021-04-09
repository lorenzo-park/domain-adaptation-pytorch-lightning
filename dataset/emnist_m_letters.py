from __future__ import print_function

import errno
import os
import numpy as np
import torch
import torch.utils.data as data
import pickle as pkl
from PIL import Image


class EMNISTMLetters(data.Dataset):
    """`EMNIST-M letters Dataset."""

    url = "https://github.com/lorenzo-park/emnist-m/releases/download/1.2/emnistm_letters_train.pkl"

    def __init__(self, root, mnist_root="data", transform=None, target_transform=None, download=False):
        """Init EMNIST-M letters dataset."""
        super(EMNISTMLetters, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.training_file = 'emnistm_letters_train.pkl'

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")
    
        data = pkl.load(open(os.path.join(self.root, self.training_file), 'rb'))
        self.train_data = data["data"]
        self.train_target = data["target"]

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.train_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file))

    def download(self):
        """Download the EMNIST-M letters data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # download pkl files
        print("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, filename)
        data = urllib.request.urlopen(self.url)
        with open(file_path, "wb") as f:
            f.write(data.read())