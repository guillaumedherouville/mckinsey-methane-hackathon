import os
import glob
from typing import Union, Tuple

import cv2
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Compose

from .mapping import CLASS_LABEL_MAP


# Normalization and type casting.
BASE_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda image: torch.from_numpy(image / 65535).unsqueeze(0).repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Geometric transformations for an augmentation.
BASE_AUGMENTATION = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ElasticTransform(alpha=20.0, sigma=2.0),
    ]), p=0.5)
])


class PlumesDataset(Dataset):
    """PyTorch dataset to load and transform plumes images and related target labels.

    @param: root_dir: The root directory, where the data is located. Must contain sub-dirs 'train' and 'test'.
    @param: is_train: The flag, signifying if to load 'train' set. If false, loads 'test' set.
    @param transform: The PyTorch transform applied to the image. By default, is set to the 'BASE_TRANSFORM'.
    @param augment: The flag, signifying if to augment image. Is applied independently of the 'transform'.
    """

    def __init__(self,
                 root_dir: str, is_train: bool = True, transform: Union[Compose, Module] = None, augment: bool = False):
        super().__init__()
        sub_dir = "train" if is_train else "test"

        self.transform = transform if transform else BASE_TRANSFORM
        self.augment = augment

        self.folders = list(CLASS_LABEL_MAP.keys())
        self.image_paths = list()
        self.image_labels = list()

        # Prefetch paths to images of all classes.
        for label, img_class in enumerate(self.folders):
            class_images_paths = glob.glob(os.path.join(root_dir, sub_dir, img_class, "*.tif"))
            self.image_paths.extend(class_images_paths)
            self.image_labels.extend([float(label)] * len(class_images_paths))

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        @return: The number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float]:
        """Get the sample from the dataset by the index.

        @param idx: Index of the sample in the dataset.
        @return: Transformed (+ augmented) image and related target value.
        """
        label = self.image_labels[idx]
        path = self.image_paths[idx]

        # Reading an image as it is.
        img = cv2.imread(path, -1)

        if self.transform:
            img = self.transform(img)

        if self.augment:
            img = BASE_AUGMENTATION(img)

        return img, label
