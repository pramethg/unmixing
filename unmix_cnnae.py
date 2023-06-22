import os, sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath('./src'))
from models.autoencoder import *
from datasets import *

if __name__ == "__main__":
    batch_size, lrate, epochs = 6, 3e-4, 300
    wave_list, depths = np.arange(700, 981, 10), np.arange(15, 41, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = MultipleCholesterolDataset(
                                root = "./data",
                                wavelist = wave_list,
                                depths = depths,
                                transform = transforms.ToTensor()
                                )
    train_loader = DataLoader(
                                dataset = train_data,
                                batch_size = batch_size,
                                shuffle = True
                                )