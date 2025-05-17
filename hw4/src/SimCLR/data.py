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

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from prepare_data import get_test_train_data, read_images
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
        image = (image * 255).astype(np.uint8)
        label = self.y_data[item]

        # TODO: pass your code
        x1 = self.transform_augment(image=image)['image']
        x2 = self.transform_augment(image=image)['image']

        image = torch.tensor(image).permute(2, 0, 1)
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
    # X_train, y_train = read_images('/../../images_background')
    X_train, y_train = read_images(path)
    # X_test, y_test = read_images('../images_evaluation')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    MEAN = np.mean(X_train, axis=(0, 1, 2), keepdims=True).squeeze()
    STD = np.std(X_train, axis=(0, 1, 2), keepdims=True).squeeze()

    train_transform = A.Compose([
        A.OneOf([
            A.ColorJitter(),
            A.ToGray(),

        ]),
        A.HorizontalFlip(),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    train_dataset, valid_dataset = load_datasets(X_train, y_train, X_val, y_val, train_transform, valid_transform, crop_coef=1.4)
    # train_dataset, test_dataset = load_datasets(X_train, y_train, X_test, y_test, train_transform, valid_transform, crop_coef=1.4)
    test_dataset = []

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    get_datasets()