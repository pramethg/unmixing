import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath('./models'))
from utils import *
from datasets import *
from models.autoencoder import *

if __name__ == "__main__":
    batch_size, lrate, epochs = 1, 1e-3, 200
    wave_list, depth, ncomp = np.arange(700, 981, 10), 25, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = SingleCholesterolDataset(
                                root = "./data/hb_hbo2_fat_29",
                                wavelist = wave_list,
                                depth = depth,
                                transform = None
                                )
    train_loader = DataLoader(
                                dataset = train_data,
                                batch_size = batch_size,
                                shuffle = True
                                )
    
    model = Autoencoder(len(wave_list), 3).to(device = device)
    model = model.float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lrate)

    losses = []
    model.train()
    progressbar = tqdm(range(epochs))
    for epoch in range(epochs):
        epoch_loss = []
        for batchidx, data in enumerate(train_loader):
            data = data.to(device = device).float()
            pred = model(data.float())
            loss = criterion(pred, data)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(mean_loss)
        progressbar.update(1)
        progressbar.set_postfix_str(s = f"MSE Loss: {mean_loss:.5f}")
    
    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_15/PA_Image_{wave}.mat')['Image_PA']) for wave in wave_list])
    c, h, w = sim_data.shape
    sim_data = torch.tensor(np.expand_dims(sim_data.transpose((1, 2, 0)).reshape((h*w, c)), axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].reshape((h, w, ncomp))
    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    plot_3d_multiple(Y*1000, X*1000, preds, title = None, cmap = 'jet', clim = None)

'''
    plt.imshow(np.array(train_data[0][:,0].reshape((396, 101))), cmap = "hot")
    plt.colorbar()
    plt.show()
'''