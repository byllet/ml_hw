import random

import numpy as np

import torch
import torch.nn as nn


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed()


class EncoderModule(nn.Module):
    def __init__(self, device, in_channels, out_channels=64, conv_kernel_size=3):
        super(EncoderModule, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels,
                                    kernel_size=conv_kernel_size,
                                    device=device,
                                    padding=conv_kernel_size-2)
        self.norm_layer = nn.BatchNorm2d(num_features=64,
                                    device=device)
        self.relu_layer = nn.ReLU()
        self.max_pool_layer = nn.MaxPool2d((2, 2), ceil_mode=False)
        

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.relu_layer(x)
        x = self.max_pool_layer(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, device, x_dim, hid_dim, z_dim):
        super(Encoder, self).__init__()
        self.module_list = nn.ModuleList([EncoderModule(device, x_dim[0]) if i == 0 
                                      else EncoderModule(device, in_channels=z_dim)
                                      for i in range(hid_dim)])

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
    
    
