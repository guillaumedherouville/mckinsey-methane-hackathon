import os
import glob

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .mapping import CLASS_LABEL_MAP

# Normalization of an uint16 image.
_BASE_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda image: torch.from_numpy(image / 65535).unsqueeze(0).repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_BASE_AUGMENTATION = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ElasticTransform(alpha=20.0, sigma=2.0),
    ]), p=0.5)
])


class PlumesDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None, augment=False):
        # Custom 'ToTensor' transform for the uint16 tiff images.
        self.is_train = is_train
        if self.is_train:
            sub_dir = "train"
        else:
            sub_dir = "test"

        self.transform = transform if transform else _BASE_TRANSFORM

        self.augment = augment
        self.augmentation_transform = _BASE_AUGMENTATION

        self.folders = list(CLASS_LABEL_MAP.keys())
        self.image_paths = list()
        self.image_labels = list()

        for label, img_class in enumerate(self.folders):
            class_images_paths = glob.glob(os.path.join(root_dir, sub_dir, img_class, "*.tif"))
            self.image_paths.extend(class_images_paths)
            self.image_labels.extend([float(label)] * len(class_images_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        path = self.image_paths[idx]

        # Reading an image as it is.
        img = cv2.imread(path, -1)

        if self.transform:
            img = self.transform(img)

        if self.augment:
            img = self.augmentation_transform(img)

        return img, label
