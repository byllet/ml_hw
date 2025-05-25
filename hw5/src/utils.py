import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_std_mean(X):
    MEAN = np.mean(X, axis=(0, 1, 2, 3), keepdims=True).squeeze()
    STD = np.std(X, axis=(0, 1, 2, 3), keepdims=True).squeeze()
    return STD, MEAN

def get_test_transform(STD, MEAN):
    transform = A.Compose([
        A.Resize(height=160, width=160),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    return transform

def get_train_transform(STD, MEAN):
    transform = A.Compose([
        A.Resize(height=160, width=160),
        A.OneOf([
            A.ColorJitter(),
            A.ToGray(),
            A.ChannelDropout(p=0.2),
        ]),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.15),
                        hole_width_range=(0.1, 0.15), p=0.5),
        A.OneOf([
            A.Affine(rotate=(-15, 15), p=0.7), 
            A.HorizontalFlip()
        ]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    return transform