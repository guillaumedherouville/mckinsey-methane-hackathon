import os
import glob

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .mapping import CLASS_LABEL_MAP

# Normalization of an uint16 image.
_BASE_TRANSFORM = transforms.Lambda(lambda image: torch.from_numpy(image / 65535).unsqueeze(0))


class CustomDataset(Dataset):
    def __init__(self, root_dir, n_channels=1, transform=None):
        # Custom 'ToTensor' transform for the uint16 tiff images.
        self.n_channels = n_channels
        self.transform = transform if transform else _BASE_TRANSFORM

        self.folders = list(CLASS_LABEL_MAP.keys())
        self.image_paths = list()
        self.image_labels = list()

        for label, img_class in enumerate(self.folders):
            class_images_paths = glob.glob(os.path.join(root_dir, img_class, "*.tif"))
            self.image_paths.extend(class_images_paths)
            self.image_labels.extend([label] * len(class_images_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        path = self.image_paths[idx]

        # Reading an image as it is.
        img = cv2.imread(path, -1)

        if self.transform:
            img = torch.Tensor(self.transform(img))

        img = img.repeat(self.n_channels, 1, 1)

        return img, label
