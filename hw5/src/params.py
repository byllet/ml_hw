import torch

hyp = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 30,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = './hw5/data/'