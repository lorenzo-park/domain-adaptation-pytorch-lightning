"""
CREDIT: https://github.com/thuml/Transfer-Learning-Library/tree/dev/dalib/vision/datasets
"""
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets.utils import download_and_extract_archive

from typing import Optional
import os
import pandas as pd
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31(ImageList):
    """Office31 Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/1f5646f39aeb4d7389b9/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/05640442cd904c39ad60/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/a069d889628d4b468c32/?dl=1"),
        ("webcam", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/4c4afebf51384cf1aa95/?dl=1"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt"
    }
    image_list_train = {
        "A": "image_list/amazon.train",
        "D": "image_list/dslr.train",
        "W": "image_list/webcam.train"
    }
    image_list_test = {
        "A": "image_list/amazon.test",
        "D": "image_list/dslr.test",
        "W": "image_list/webcam.test"
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, train: bool, download: Optional[bool] = True, **kwargs):
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
#         self.__split__(root)
        
#         if train:
#             self.image_list = self.image_list_train
#         else:
#             self.image_list = self.image_list_test

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(Office31, self).__init__(root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)

        
    def __split__(self, root: str):
        domains = list(filter(lambda x: ".txt" in x, os.listdir(root)))
        samples = []
        for domain in domains:
            with open(os.path.join(root, domain), "r") as f:
                for line in f.readlines():
                    file_path, label = line.strip().split()
                    samples.append({
                        "path": file_path,
                        "label": label,
                        "domain": domain.replace(".txt", ""),
                    })
        df = pd.DataFrame(samples)
        df["label_domain"] = df["label"] +df["domain"]
        skf = StratifiedKFold(n_splits=5)
        df["fold"] = -1
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(df, df["label_domain"])):
            df.iloc[val_idx, -1] = fold_id
        test_df = df[(df["fold"] == 0)]
        train_df = df.drop(test_df.index)
        
        for domain, _, _ in self.download_list:
            if domain == "image_list":
                continue
            try:
                os.remove(os.path.join(root, "image_list", f"{domain}.train"))
                os.remove(os.path.join(root, "image_list", f"{domain}.test"))
            except OSError:
                pass
            
        for idx, row in test_df.iterrows():
            with open(os.path.join(root, "image_list", f"{row['domain']}.train"), "a") as f:
                f.write(f"{row['path']} {row['label']}\n")
        for idx, row in train_df.iterrows():
            with open(os.path.join(root, "image_list", f"{row['domain']}.test"), "a") as f:
                f.write(f"{row['path']} {row['label']}\n")