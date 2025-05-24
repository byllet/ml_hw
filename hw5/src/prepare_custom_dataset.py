import torch
import numpy as np
from torch.utils.data import  DataLoader, WeightedRandomSampler
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import SiameseNetwork
from params import DEVICE, DATA
import itertools
import random
import os

from evaluate import predict_on_n_pairs
from data import get_data, LFWDataset

def rename_images(dir_path, transform=False): 
    if transform:
        for name in ["Anton", "Rasim", "Eva"]:
            for i, filename in enumerate(os.listdir(os.path.join(dir_path, name))):
                os.rename(os.path.join(dir_path, name, filename), 
                          os.path.join(dir_path, name, f"{name}_000{i+1}.jpg"))  
                
def choose_one_image(files_pathes):
    suitable_dirs = [dir for dir in files_pathes if len(dir) >= 6]
    suitable_dirs = list(itertools.chain(*suitable_dirs))
    return random.choice(suitable_dirs)

def create_custom_dateset(path):
    files_pathes = []
    for dir_name in os.listdir(path):
        dir_files_pathes = [os.path.join(path, dir_name, file) for file in os.listdir(os.path.join(path, dir_name))]
        files_pathes.append(dir_files_pathes)
    
    image_to_compare = choose_one_image(files_pathes)
    names_for_dataset = []
    for dir_name in os.listdir(path):
        if dir_name in image_to_compare:
           names_for_dataset += [(image_to_compare, os.path.join(path, dir_name, file), 1) \
                        for file in os.listdir(os.path.join(path, dir_name)) \
                        if image_to_compare != os.path.join(path, dir_name, file)]
        else: 
            names_for_dataset += [(image_to_compare, os.path.join(path, dir_name, file), 0) \
                        for file in os.listdir(os.path.join(path, dir_name))]

    X = []
    y = []
    for name1, name2, is_similar in names_for_dataset:
        img1 = Image.open(name1)
        img2 = Image.open(name2)
        newsize = (160, 160)
        X.append((np.asarray(img1.resize(newsize)), np.asarray(img2.resize(newsize))))
        y.append(is_similar)
        img1.close()
        img2.close()
    return np.asarray(X), np.asarray(y)
        

def main(model_path, out_dir, batch_size=15):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))

    X_train, _, _, _ = get_data(DATA)
    MEAN = np.mean(X_train, axis=(0, 1, 2, 3), keepdims=True).squeeze()
    STD = np.std(X_train, axis=(0, 1, 2, 3), keepdims=True).squeeze()

    transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    X_test, y_test = create_custom_dateset(os.path.join(DATA, "custom_dataset"))
    valid_dataset = LFWDataset(X_test, y_test, transform_augment=transform)
    sampler = WeightedRandomSampler([5/batch_size if y == 1 else 10/batch_size for y in y_test], batch_size, replacement=False)
    test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    predict_on_n_pairs(model, out_dir, DEVICE, test_loader, MEAN, STD, batch_size)

if __name__ == "__main__":
    rename_images(os.path.join(DATA, "custom_dataset/"))
    main("hw5/data/model_on_250.pth", "hw5/eval_on_custom_dataset")