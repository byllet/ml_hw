import requests
import zipfile
import os

def load_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def extract(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_and_extract(url, path_to):
    zip_path = 'temp.zip'
    load_file(url, zip_path)
    extract(zip_path, path_to)
    os.remove(zip_path)

if __name__=="__main__":
    load_and_extract('https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip', '.')
    load_and_extract('https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip', '.')
