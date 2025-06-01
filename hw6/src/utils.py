import torch
import cv2
import numpy as np
import os
from PIL import Image

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor


def transform_image(image):
    background = get_background(image)
    image1 = image.astype(np.float32, copy=True)
    difference = abs(image1 - background)
    difference = difference[0:288, 0:404]
    image_data = difference.astype(np.uint8, copy=False)
    image_data = cv2.cvtColor(cv2.resize(difference, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.resize(image_data, (84, 84, 1))
    #print(image_data[:, :, 0].shape, image_data.shape)
    #img = Image.fromarray(image_data[:, :, 0], 'L')
    #img = Image.fromarray(image_data)
    #img.show()
    #s = input()
    return image_data
    
    
def get_background(image, path="assets/sprites/"):
    back1 = np.array(Image.open(os.path.join(path, "background-day.png")).convert("RGB")).transpose((1, 0, 2))
    back2 = np.array(Image.open(os.path.join(path, "background-night.png")).convert("RGB")).transpose((1, 0, 2))
    image1 = image.astype(np.float32, copy=True)
    back1 = back1.astype(np.float32, copy=True)
    back2 = back2.astype(np.float32, copy=True)
    if sum(abs((image1 - back1)[0][0])) == 0:
        return back1
    else:
        return back2
    
