import os
import glob

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .mapping import CLASS_LABEL_MAP


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        # Custom 'ToTensor' transform for the uint15 tiff images.
        self.transform = transforms.Lambda(lambda image: torch.from_numpy(image / 65535))

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
            img = self.transform(img)

        return img, label

