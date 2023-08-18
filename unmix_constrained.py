import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
import numpy as np
from tqdm import trange
from datasets import ZCA
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datasets import SingleCholesterolDataset
warnings.filterwarnings('ignore')

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation = 'trelu'):
        super(AutoEncoder, self).__init__()
        self.encw = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.encb = nn.Parameter(torch.zeros(hidden_dim))
        self.decw = self.encw.t()
        self.decb = nn.Parameter(torch.zeros(input_dim))

        self.alpha = nn.Parameter(torch.randn(hidden_dim))
        self.beta = nn.Parameter(torch.randn(hidden_dim))

        activations = {'trelu': self.trelu, 'tsigmoid': self.tsigmoid, 'tsoftplus': self.tsoftplus, 'sigmoid': F.sigmoid, 'relu': F.relu, 'softplus': F.softplus, 'tanh': F.tanh}
        assert activation in activations, 'Pick Valid Activation Function'
        self.activation = activations[activation]
    
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
        decout = F.sigmoid(torch.matmul(x, self.encw.t()) + self.decb)
        return encout, decout    

def test():
    x = torch.randn(2048, 10)
    model = AutoEncoder(10, 3, 'tsigmoid')
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

def unmixing(save = False, seed = 9, beta = 0.5, zcatest = True):
    BATCH_SIZE = 396*101
    EPOCHS = 300
    LRATE = 3e-4
    NCOMP = 3
    wavelist = []
    torch.manual_seed(seed = seed)
    data = SingleCholesterolDataset(root = './data/hb_hbo2_fat_11_20', wavelist = [], depth = [20], whiten = 'zca', normalize = True)
    dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = False, num_workers = 16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(len(data.wavelist), NCOMP).to(device = device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LRATE)
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
            negentropy = -(torch.mean(-torch.exp(-(encoded ** 2) / 2)) - torch.mean(-torch.exp(-(torch.randn_like(encoded)) ** 2 / 2)))
            loss = (beta * mse) + ((1 - beta) * negentropy)
            loss.backward()
            optimizer.step()
            epochloss.append(loss)
            t.set_description_str(f'EPOCH: [{epoch + 1}/{EPOCHS}]')
            t.set_postfix_str(f'LOSS: {loss.item()}')
        losses.append(sum(epochloss) / len(epochloss))
    
    if save:
        torch.save(model.state_dict(), 'CONSTRAINED.pth')
    
    plt.figure(figsize = (8, 5))
    plt.plot(list(range(EPOCHS)), losses)
    plt.savefig('./CONSTRAINED/Loss.png', dpi = 500)
    plt.close()

    sim_data = np.array([np.array(loadmat(f'./data/hb_hbo2_fat_29_11/PA_Image_{wave}.mat')['Image_PA']) for wave in wavelist])
    c, h, w = sim_data.shape
    sim_data = sim_data.transpose((1, 2, 0)).reshape((h*w, c))
    zca = ZCA()
    sim_data = ZCA
    sim_data = torch.tensor(np.expand_dims(sim_data, axis = 0), dtype = torch.float32)
    preds = np.array(model.encoder(sim_data.to(device)).cpu().detach())[0].reshape((h, w, NCOMP))
    
    wts = (model.encoder[0].weight).cpu().detach().numpy()
    plot_comps_2d(preds, wave_list, np.linalg.pinv(wts), order = [0, 1, 2], xticks = wave_list, title = 'FCAE', save = 'fcae')

if __name__ == "__main__":
    test()