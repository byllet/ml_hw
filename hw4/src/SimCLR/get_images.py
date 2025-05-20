import cv2
import numpy as np
import torch
import os


def read_alphabets(alphabet_directory_path):
    """
    Reads all the characters from a given alphabet_directory
    Args:
      alphabet_directory_path (str): path to diretory with files
    Returns:
      datax (np.array): array of path name of images
      datay (np.array): array of labels
    """
    datax = []  # all file names of images
    datay = []  # all class names f

    alphabet_name = alphabet_directory_path.split('/')[-2]
    
    for character in os.listdir(alphabet_directory_path):
        class_name = f"{alphabet_name}_{character}"

        character_path = alphabet_directory_path +  character
        for image_name in os.listdir(character_path):
            image = cv2.resize(cv2.imread(character_path + '/' + image_name), (28, 28))
            datay.append(class_name)
            datax.append(image)  
                
    return np.array(datax), np.array(datay)


def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    
    results = [read_alphabets(base_directory + '/' + directory + '/') for directory in os.listdir(base_directory)]
    
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.concatenate([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    
    return datax, datay
