import warnings
warnings.filterwarnings('ignore')
import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath('./src'))
from utils import *
from datasets import *
from models.autoencoder import *

if __name__ == "__main__":
    batch_size, lrate, epochs = 6, 3e-4, 300
    wave_list, depths, ncomp = np.arange(700, 981, 10), np.arange(25, 41, 5), 3
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
    
    model = ConvAutoencoder(
                                in_channels = len(wave_list),
                                hidden_channels = ncomp,
                                ).to(device = device)
    model = model.float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lrate)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = []
        for batch, data in enumerate(train_loader):
            data = data.to(device)
            pred = model(data.float())
            loss = criterion(pred, data)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(mean_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1} - Loss: {mean_loss:.5f}")

    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_15/PA_Image_{wave}.mat')['Image_PA']) for wave in wave_list])
    c, h, w = sim_data.shape
    sim_data = torch.tensor(np.expand_dims(sim_data, axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].transpose((1, 2, 0))
    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    plot_3d_multiple(Y*1000, X*1000, preds, title = None, cmap = 'jet', clim = None)