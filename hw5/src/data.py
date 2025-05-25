import torch
from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils import get_std_mean, get_test_transform, get_train_transform


class LFWDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data

        assert transform_augment is not None, 'set transform_augment'
        self.transform_augment = transform_augment

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        x1 = self.x_data[item, 0]
        x1 = (x1 * 255).astype(np.uint8)
        x2 = self.x_data[item, 1]
        x2 = (x2 * 255).astype(np.uint8)

        x1 = self.transform_augment(image=x1)['image']
        x2 = self.transform_augment(image=x2)['image']

        label = self.y_data[item]
        label = torch.tensor(label, dtype=torch.long)

        return x1, x2, label


def get_data(data_path):
    lfw_dataset = fetch_lfw_pairs(data_home=data_path, color=True, resize=0.9, slice_=(slice(1, 249), slice(1, 249)), download_if_missing=True)
    X_train, X_test, y_train, y_test = train_test_split(
        lfw_dataset.pairs, lfw_dataset.target, test_size=0.25, stratify=lfw_dataset.target, random_state=42)
    
    return X_train, X_test, y_train, y_test


def get_train_test_dataloader(data_path):
    X_train, X_test, y_train, y_test = get_data(data_path)
    
    STD, MEAN = get_std_mean(X_train)
    
    train_transform = get_train_transform(STD, MEAN)
    valid_transform = get_test_transform(STD, MEAN)
    
    train_dataset = LFWDataset(X_train, y_train, transform_augment=train_transform)
    valid_dataset = LFWDataset(X_test, y_test, transform_augment=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    return train_loader, valid_loader
