import random
from pathlib import Path
from time import gmtime, strftime

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from get_images import read_images


class CLDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data

        unique_labels = sorted(set(y_data))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.y_data = [self.label2id[label] for label in y_data]

        assert transform_augment is not None, 'set transform_augment'
        # TODO: pass your code
        self.transform_augment = transform_augment

    def __len__(self):
        # TODO: pass your code
        return len(self.x_data)

    def __getitem__(self, item):
        image = self.x_data[item]
        # image = (image * 255)
        label = self.y_data[item]

        # TODO: pass your code
        x1 = self.transform_augment(image=image)['image'].float()
        x2 = self.transform_augment(image=image)['image'].float()

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)
        return x1, x2, label, image
    

def get_cropped_data_idxs(data, crop_coef: float = 1.0):
    crop_coef = np.clip(crop_coef, 0, 1)

    init_data_size = len(data)
    final_data_size = int(init_data_size * crop_coef)

    random_idxs = np.random.choice(tuple(range(init_data_size)), final_data_size, replace=False)
    return random_idxs
    

def load_datasets(X_train, y_train, X_val, y_val, train_transform, valid_transform, crop_coef=0.2):
    train_idxs = get_cropped_data_idxs(X_train, crop_coef=crop_coef)
    train_data = X_train[train_idxs]
    train_labels = y_train[train_idxs]

    valid_idxs = get_cropped_data_idxs(X_val, crop_coef=crop_coef)
    valid_data = X_val[valid_idxs]
    valid_labels = y_val[valid_idxs]


    train_dataset = CLDataset(train_data, train_labels, transform_augment=train_transform)
    valid_dataset = CLDataset(valid_data, valid_labels, transform_augment=valid_transform)

    return train_dataset, valid_dataset


def get_datasets(path):
    X_train, y_train = read_images(path)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.35, random_state=42, shuffle=True)

    train_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(p=0.3),
        ], p=0.5),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        ToTensorV2()
    ])

    train_dataset, valid_dataset = load_datasets(X_train, y_train, X_val, y_val, train_transform, valid_transform, crop_coef=1.4)

    return train_dataset, valid_dataset


if __name__ == "__main__":
    pass