import os
from typing import Any
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SimulatedFC(Dataset):
    def __init__(self, root, transform = None):
        super(SimulatedFC, self).__init__()
        self.root = root
        self.transform = transform
    
    def __len__(self):
        pass