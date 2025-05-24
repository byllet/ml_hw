import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import  DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse
import os
import json

from model import SiameseNetwork
from params import DEVICE, DATA

from data import get_data, LFWDataset

def predict_on_n_pairs(model, output_dir, device, test_loader,  MEAN, STD, n = 10):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    count = 0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    accuracy = 0
    TP = 0
    TN = 0

    with torch.no_grad():
        for x1, x2, labels in test_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)
            preds = model(x1, x2).squeeze()

            for i in range(x1.size(0)):
                if count >= n:
                    break
                count += 1

                sim_score = preds[i].item()
                match = sim_score > 0.5

                accuracy += match == labels[i].item()
                TP += (match == labels[i].item()) and match
                TN += (match == labels[i].item()) and not match

                pair_dir = os.path.join(output_dir, f'pair_{count}')
                os.makedirs(pair_dir, exist_ok=True)

                to_pil_image((x1[i].cpu() * std + mean).clamp(0, 1)).save(os.path.join(pair_dir, 'img1.jpg'))
                to_pil_image((x2[i].cpu() * std + mean).clamp(0, 1)).save(os.path.join(pair_dir, 'img2.jpg'))

                with open(os.path.join(pair_dir, 'result.txt'), 'w') as f:
                    f.write(f"Similarity score: {sim_score:.4f}\n")
                    f.write(f"Predicted match: {'Yes' if match else 'No'}\n")
                    f.write(f"True label: {int(labels[i].item())}\n")


    with open(os.path.join(output_dir, 'accuracy.json'), 'w') as f:
        json.dump({"accuracy": accuracy / n, 
                   "TP": round(TP / (sum([i.item() for i in labels]) + 1e-6), 2), 
                   "TN": round(TN / (sum([(1 - i.item()) for i in labels]) + 1e-6), 2),
                   "total similar": int(sum([i.item() for i in labels])),
                   "total different": int(sum([1 - i.item() for i in labels]))}, f)
        


def main(model_path, out_dir):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))


    X_train, X_test, y_train, y_test = get_data(DATA)
    MEAN = np.mean(X_train, axis=(0, 1, 2, 3), keepdims=True).squeeze()
    STD = np.std(X_train, axis=(0, 1, 2, 3), keepdims=True).squeeze()

    transform = A.Compose([
        A.Resize(height=160, width=160),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    valid_dataset = LFWDataset(X_test, y_test, transform_augment=transform)
    test_loader = DataLoader(valid_dataset, batch_size=10, shuffle=True)

    predict_on_n_pairs(model, out_dir, DEVICE, test_loader, MEAN, STD,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py")
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--out_path",
        type=str,
    )

    args = parser.parse_args()
    main(args.model_path, args.out_path)

#python .\hw5\src\evaluate.py --model_path .\hw5\hw5siam.pth --out_path .\hw5\eval_on_test