from model import SiameseNetwork
from params import DEVICE
import torch
from PIL import Image
from torchvision import transforms
import argparse 
from utils import get_test_transform
import numpy as np

def main(model_path, path_to_img1, path_to_img2):
    STD = [0.31974083, 0.31662673, 0.2981077 ]
    MEAN = [0.10223421, 0.10223421, 0.10223421]
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    transform = get_test_transform(STD, MEAN)
    img1 = np.array(Image.open(path_to_img1))
    transformed = transform(image=img1)
    img1 = transformed["image"].unsqueeze(0).to(DEVICE)
    img2 = np.array(Image.open(path_to_img2))
    transformed = transform(image=img2)
    img2 = transformed["image"].unsqueeze(0).to(DEVICE)
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        sim_score = model(img1, img2).squeeze().item()
        print(f"Similarity Score: {sim_score:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="launch_model.py")
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--img1_path",
        type=str,
    )
    parser.add_argument(
        "--img2_path",
        type=str,
    )

    args = parser.parse_args()
    main(args.model_path, args.img1_path, args.img2_path)
#python src/launch_model.py --model_path model_on_250.pth --img1_path data/custom_dataset/Rasim/Rasim_0001.jpg --img2_path data/custom_dataset/Anton/Anton_0005.jpg