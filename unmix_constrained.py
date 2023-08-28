import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
import numpy as np
from tqdm import trange
from scipy.io import loadmat
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from utils import *
from datasets import ZCA
from datasets import SingleCholesterolDataset

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation = 'trelu', tied = True, winit = False, wdecay = 1e-5):
        super(AutoEncoder, self).__init__()
        self.tied = tied
        self.wdecay = wdecay
        encw = wtscale(symdecorrelation(torch.randn(input_dim, hidden_dim))) if winit else torch.randn(input_dim, hidden_dim)
        self.encw = nn.Parameter(encw)
        self.encb = nn.Parameter(torch.zeros(hidden_dim))
        self.decw = self.encw.t() if self.tied else nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.decb = nn.Parameter(torch.zeros(input_dim))

        if activation[0] == 't':
            self.alpha = nn.Parameter(torch.randn(hidden_dim))
            self.beta = nn.Parameter(torch.randn(hidden_dim))

        activations = {'trelu': self.trelu, 'tsigmoid': self.tsigmoid, 'tsoftplus': self.tsoftplus, 'sigmoid': F.sigmoid, 'relu': F.relu, 'softplus': F.softplus, 'tanh': F.tanh}
        assert activation in activations, 'Pick Valid Activation Function'
        self.activation = activations[activation]
    
    def l2regularization(self):
        return self.wdecay * torch.sum(self.encw ** 2)
    
    def tsoftplus(self, x):
        return F.softplus(self.beta + (self.alpha * x))
    
    def tsigmoid(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))
    
    def trelu(self, x):
        return torch.relu(self.alpha * x + self.beta)

    def encode(self, x):
        return self.activation(torch.matmul(x, self.encw) + self.encb)

    def forward(self, x):
        x = torch.matmul(x, self.encw) + self.encb
        encout = self.activation(x)
        decout = F.sigmoid(torch.matmul(encout, self.encw.t()) + self.decb) if self.tied \
                else F.sigmoid(torch.matmul(encout, self.decw) + self.decb)
        return encout, decout
        
    def sparsity(self, activations):
        mean = torch.mean(activations, dim = 0)
        mean = torch.clamp(mean, self.eps, 1 - self.eps)
        kldiv = (self.sparsity_target * torch.log(self.sparsity_target / mean)) + ((1 - self.sparsity_target) *  torch.log((1 - self.sparsity_target) / (1 - mean)))
        return torch.sum(kldiv)

def test(tied = True):
    x = torch.randn(2048, 10)
    model = AutoEncoder(10, 3, 'tsigmoid', tied)
    model.eval()
    assert model.encw.all() == model.decw.t().all()
    print(model.forward(x)[0].shape)
    print(model.forward(x)[1].shape)
    
    print('Parameters: ')
    for idx, param in enumerate(model.parameters()):
        print(f'{idx + 1}: {param.shape}')

def symdecorrelation(W):
    S, U = torch.linalg.eigh(torch.matmul(W, W.T))
    S = torch.clip(S, min = torch.finfo(W.dtype).tiny, max = None)
    return torch.linalg.multi_dot([U * (1.0 / (torch.sqrt(S) + 1e-12)), U.T, W])

def icaupdate(W, encoded):
    Wgrad = torch.eye(W.size(1)) + (1 - 2 * torch.tanh(encoded)).t() @ W
    W = W + (0.01 * Wgrad)
    W = symdecorrelation(W)
    return W

def wtscale(W):
    W = (W - W.min()) / (W.max() - W.min())
    Wsum = torch.sum(W, axis = 1)
    for idx in range(W.shape[0]):
        W[idx] /= Wsum[idx]
    return W

def unmixing(seed = 9, beta = 1, negexp = 1.0, l2reg = True, save = False, wnorm = False):
    BATCH_SIZE = 396*101
    EPOCHS = 150
    LRATE = 7e-3
    NCOMP = 3
    SEED = 9
    negexp = 1.2
    np.random.seed(seed = seed)
    torch.manual_seed(seed = SEED)
    data = SingleCholesterolDataset(root = './data/hb_hbo2_fat_11', wavelist = 'EXP10', depth = [20], whiten = 'zca', normalize = True)
    dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = False, num_workers = 16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(len(data.wavelist), NCOMP, activation = 'sigmoid', tied = False, winit = True).to(device = device)
    print(f'Model Parameters: {sum(param.numel() for param in model.parameters())}')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LRATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 10, verbose = True)
    losses = []
    model.train()
    for epoch in (t := trange(EPOCHS)):
        epochloss = []
        for batch in dataloader:
            epochloss = []
            batch = batch.to(device)
            optimizer.zero_grad()
            encoded, decoded = model.forward(batch)
            mse = criterion(decoded, batch)
            negentropy = torch.abs(-(torch.mean(-torch.exp(- negexp * (encoded ** 2) / 2)) - torch.mean(-torch.exp(- negexp * (torch.randn_like(encoded)) ** 2 / 2))))
            loss = ((beta * mse) + ((1 - beta) * negentropy)) if beta != 1 else (mse + negentropy)
            if l2reg:
                loss += model.l2regularization()
            if wnorm:
                with torch.no_grad():
                    model.encw.data = symdecorrelation(model.encw.data)
                    minval, maxval = model.encw.data.min(), model.encw.data.max()
                    model.encw.data = (model.encw.data - minval) / (maxval - minval + 1e-12)
            loss.backward()
            optimizer.step()
            epochloss.append(loss.item())
            t.update(1)
            t.set_description_str(f'EPOCH: [{epoch + 1}/{EPOCHS}]')
            t.set_postfix_str(f'MSELOSS: {mse.item():.3f} NEGENTROPY: {negentropy.item():.3f} TOTAL LOSS: {loss.item():.3f}')
        epochlossmean = sum(epochloss) / len(epochloss)
        scheduler.step(epochlossmean)
        losses.append(epochlossmean)
        if epochlossmean < 0.02:
            break
    
    if save:
        torch.save(model.state_dict(), 'CONSTRAINED.pth')
    
    plt.figure(figsize = (8, 5))
    plt.plot(list(range(len(losses))), losses)
    plt.savefig('./CONSTRAINED/Loss.png', dpi = 500)
    plt.close()

    testdata = SingleCholesterolDataset(root = './data/hb_hbo2_fat_11', wavelist = 'EXP10', depth = [20], whiten = 'zca', normalize = True)
    preds = np.array(model.encode(testdata[0].to(device)).cpu().detach()).reshape((396, 101, NCOMP))
    
    plot_comps_2d(preds, data.wavelist, model.encw.detach().cpu().numpy(), order = [0, 1, 2], xticks = data.wavelist, title = 'CONSTRAINED AE', save = './CONSTRAINED/Encoder')

    simdata = testdata[0].numpy().reshape((396, 101, len(testdata.wavelist)))
    decout = model.forward(testdata[0].to(device))[1].detach().cpu().numpy().reshape((396, 101, len(testdata.wavelist)))
    plt.figure(figsize = (12, 4))
    for idx in range(len(testdata.wavelist)):
        plt.subplot(2, 10, idx + 1)
        plt.imshow(simdata[:, :, idx], cmap = "hot")
        plt.colorbar()
        plt.subplot(2, 10, idx + 11)
        plt.imshow(decout[:, :, idx], cmap = "hot")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('./CONSTRAINED/Decoder.png')

if __name__ == "__main__":
    unmixing(seed = 9, beta = 1, negexp = 1.2, l2reg = False, save = False, wnorm = False)

"""
    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_11_20/PA_Image_{wave}.mat')['Image_PA']) for wave in data.wavelist])
    c, h, w = sim_data.shape
    sim_data = sim_data.transpose((1, 2, 0)).reshape((h*w, c))
    zca = ZCA()
    sim_data = zca.fit_transform(sim_data)
    sim_data = torch.tensor(np.expand_dims(sim_data, axis = 0), dtype = torch.float32)
"""
