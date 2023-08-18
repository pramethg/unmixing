import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
from tqdm import trange
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
    
    for param in model.parameters():
        print(param.shape)

def unmixing(save = False):
    BATCH_SIZE = 396*101
    EPOCHS = 300
    LRATE = 3e-4
    NCOMP = 3
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
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion()
        
        losses.append(sum(epochloss) / len(epochloss))
        t.set_description_str(f'EPOCH: [{epoch + 1}/{EPOCHS}]')
        t.set_postfix_str(f'LOSS: {loss.item()}')
    
    if save:
        torch.save(model.state_dict(), 'CONSTRAINED.pth')

if __name__ == "__main__":
    test()