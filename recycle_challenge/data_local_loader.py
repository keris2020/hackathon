import os

import numpy as np
import torch

from torch.utils import data
from torchvision import transforms
from PIL import Image


def get_transform():
    """Module for image pre-processing definition.

    You can customize this module.
    """
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform)


def retrieve_meta(meta_root):
    with open(meta_root, 'r') as f:
        image_ids = []
        targets = []
        for line in f.readlines():
            image_ids.append(line.strip().split(',')[0])
            str_target = line.strip().split(',')[1:]
            targets.append([int(i) for i in str_target])
    return image_ids, torch.from_numpy(np.array(targets)).type(torch.float32)


class CustomDataset(data.Dataset):
    """Dataset class.

    This class is used for internal NSML inference system.
    You can change this module for improving image load efficiency.
    """

    def __init__(self, root, transform, split):
        if split == 'test':
            self.data_root = os.path.join(root, f'{split}_data')
            self.image_ids = self.targets = \
                [img for img in os.listdir(self.data_root)]
        else:
            self.data_root = os.path.join(root, 'train/train_data')
            self.meta_root = os.path.join(self.data_root, f'{split}_label')
            self.image_ids, self.targets = retrieve_meta(self.meta_root)
        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        return image, image_id, self.targets[idx]

    def __len__(self):
        return len(self.image_ids)


def data_loader(root, split='test', batch_size=64):
    """Test data loading module.

    Args:
        root: string. dataset path.
        split: string.
        batch_size: int.

    Returns:
        DataLoader instance
    """
    input_transform = get_transform()
    dataset = CustomDataset(root, input_transform, split)
    shuffle = False if split == 'val' else True

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle)
