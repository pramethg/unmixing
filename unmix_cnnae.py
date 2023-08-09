import warnings
warnings.filterwarnings('ignore')
import os, sys
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath('./src'))
from utils import *
from datasets import *
from model.autoencoder import *

if __name__ == "__main__":
    seed = 9
    batch_size, lrate, epochs = 6, 3e-2, 300
    wave_list, depths, ncomp = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950], np.arange(15, 41, 5), 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

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
    progressbar = tqdm(range(epochs))
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
        progressbar.update(1)
        progressbar.set_postfix_str(s = f"MSE Loss: {mean_loss:.5f}")

    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_25/PA_Image_{wave}.mat')['Image_PA']) for wave in wave_list])
    c, h, w = sim_data.shape
    sim_data = torch.tensor(np.expand_dims(sim_data, axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].transpose((1, 2, 0))
    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    plot_3d_multiple(Y*1000, X*1000, preds, title = None, cmap = 'jet', clim = None)