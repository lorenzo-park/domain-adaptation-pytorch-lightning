"""
CREDIT: https://github.com/thuml/Transfer-Learning-Library/tree/dev/dalib/vision/datasets
"""

import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class StylizedDataset(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image classification

    Parameters:
        - **root** (str): Root directory of dataset
        - **classes** (List[str]): The names of all the classes
        - **data_list_file** (str): File to read the image list from.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride `parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.data = self.parse_data_file(root)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        img, target = self.data[index]
#         img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(path)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def parse_data_file(self, path: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        data_list = []
        for filename in os.listdir(path):
            if "png" in filename:
                target = int(filename.split("_")[0])
                entire_path = os.path.join(path, filename)
                img = self.loader(entire_path)
                data_list.append((img, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
