import os
import torch
import warnings
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import *
from models.nnsae import NNSAE
from datasets import SingleCholesterolDataset
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

class SimulatedCholesterolDataset(Dataset):
    def __init__(self, root = './data/hb_hbo2_fat_29', wavelist = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950], depth = 25, transform = None):
        self.root = root
        self.depth = depth
        self.wavelist = wavelist
        self.transform = transform
        self.simdata = np.array([np.array(loadmat(f'{self.root}_{self.depth}/PA_Image_{wave}.mat')['Image_PA']) for wave in self.wavelist])
        c, h, w = self.simdata.shape
        self.simdata = self.simdata.transpose((1, 2, 0)).reshape((h*w, c))
        for wave in range(len(self.wavelist)):
            self.simdata[:,wave] -= np.mean(self.simdata[:, wave])
            self.simdata[:,wave] /= np.std(self.simdata[:, wave])
    
    def __len__(self):
        return self.simdata.shape[0]
    
    def __repr__(self):
        return f"Cholesterol Dataset(Root: {self.root}_{self.depth}/, Wavelengths: {self.__len__()}, Depth: {self.depth})"
    
    def __getitem__(self, index):
        if self.transform:
            simdata = self.transform(simdata)
        return torch.Tensor(self.simdata)[index]

def validdata():
    dataset = SimulatedCholesterolDataset()
    print(len(dataset), dataset[0].shape)

def test():
    model = NNSAE(10, 3, 0.1, 0.01)
    x = torch.randn(1, 39996, 10)
    print(model(x).shape, model.encoder(x).shape)

if __name__ == "__main__":
    SEED = 1999
    SINGLE = True
    BATCH_SIZE = 1
    EPOCHS = 500
    WAVE_LIST, ncomp = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950], 3
    np.random.seed(seed = SEED)
    torch.manual_seed(seed = SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if SINGLE:
        dataset = SingleCholesterolDataset(root = './data/hb_hbo2_fat_29', 
                                        wavelist = WAVE_LIST,
                                        depth = 25, 
                                        transform = None,
                                        whiten = False,
                                        normalize = True)
    else:
        dataset = SimulatedCholesterolDataset(root = './data/hb_hbo2_fat_29', transform = None)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = NNSAE(len(dataset.wavelist), 3, 0.1, 0.01).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 3e-4)
    progressbar = tqdm(range(EPOCHS))
    losses = []
    for epoch in progressbar:
        epochloss = []
        for _, data in enumerate(dataloader):
            data = data.to(device = device)
            optimizer.zero_grad()
            for param in model.parameters():
                param.data = torch.clamp(param.data, 0)
            pred = model(data)
            loss = criterion(pred, data)
            activations = model.encoder(data)
            penalty = model.sparsity(activations)
            loss += (model.sparsity_weight * penalty)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 1.0)
            optimizer.step()
            epochloss.append(loss.item())
        meanloss = sum(epochloss) / len(epochloss)
        progressbar.update(1)
        progressbar.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        progressbar.set_postfix_str(s = f'MSE Loss: {loss.item():.3f}')
        losses.append(meanloss)
    del epochloss, data, pred, activations
    print(f'Last Epoch Loss: {losses[-1]}')

    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_25/PA_Image_{wave}.mat')['Image_PA']) for wave in WAVE_LIST])
    c, h, w = sim_data.shape
    sim_data = normalize(sim_data)
    sim_data = torch.tensor(np.expand_dims(sim_data.transpose((1, 2, 0)).reshape((h*w, c)), axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].reshape((h, w, ncomp))
    
    wts = (model.encoder.weight).cpu().detach().numpy()
    plot_comps_2d(preds, WAVE_LIST, np.linalg.pinv(wts), order = [0, 1, 2], xticks = WAVE_LIST, title = 'NNSAE', save = 'nnsae', chrom = [None]*3)