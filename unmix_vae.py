import os
import sys
import torch
import warnings
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *
from datasets import SingleCholesterolDataset
plt.style.use('dark_background')
warnings.filterwarnings('ignore')
sys.path.append('../omoikane/autoencoders/')
from vae import *

def test():
    model = VAE(10, 3, 3)
    x = torch.randn(39996, 10)
    model.eval()
    print(model(x)[0].shape, model.encoder(x).shape)

if __name__ == "__main__":
    SEED = 1999
    SINGLE = True
    RECON = 'BCE'
    BATCH_SIZE = 39996
    EPOCHS = 500
    WAVE_LIST, ncomp = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950], 3
    DEPTHS = list(range(15, 41, 5))
    np.random.seed(seed = SEED)
    torch.manual_seed(seed = SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = SingleCholesterolDataset(root = './data/hb_hbo2_fat_29', 
                                    wavelist = WAVE_LIST,
                                    depth = DEPTHS, 
                                    transform = None,
                                    whiten = False,
                                    normalize = True)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = VAE(len(dataset.wavelist), ncomp, ncomp).to(device)
    criterion = nn.MSELoss() if RECON == 'MSE' else nn.BCELoss(reduction = 'sum')
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    progressbar = tqdm(range(EPOCHS))
    losses = []
    for epoch in progressbar:
        epochloss = []
        for _, data in enumerate(dataloader):
            data = data.to(device = device)
            optimizer.zero_grad()
            pred, mu, logvar = model.forward(data)
            RECON = criterion(pred, data)
            KLD = -0.5 + torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + 1e-10)
            loss = RECON + KLD
            loss.backward()
            optimizer.step()
            epochloss.append(loss.item())
        meanloss = sum(epochloss) / len(epochloss)
        progressbar.update(1)
        progressbar.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        progressbar.set_postfix_str(s = f'RECON: {RECON.item():.3f}, KLD: {KLD.item():.3f}, TOTAL: {loss.item():.3f}')
        losses.append(meanloss)
    del epochloss, data, pred
    print(f'Last Epoch Loss: {losses[-1]}')

    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_25/PA_Image_{wave}.mat')['Image_PA']) for wave in WAVE_LIST])
    c, h, w = sim_data.shape
    sim_data = normalize(sim_data)
    sim_data = torch.tensor(np.expand_dims(sim_data.transpose((1, 2, 0)).reshape((h*w, c)), axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].reshape((h, w, ncomp))
    
    wts = (model.imghidden.weight).cpu().detach().numpy()
    plot_comps_2d(preds, WAVE_LIST, np.linalg.pinv(wts), order = [0, 1, 2], xticks = WAVE_LIST, title = 'NNSAE', save = 'nnsae', chrom = [None]*3)