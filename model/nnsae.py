import torch
import torch.nn as nn
import torch.nn.init as init
from torchinfo import summary

class NNSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, tied = False, sparsity_target = 0.1, sparsity_weight = 0.01, eps = 1e-6):
        super(NNSAE, self).__init__()
        self.eps = eps
        self.encoder = nn.Linear(input_dim, hidden_dim, bias = True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias = True)
        init.kaiming_uniform_(self.encoder.weight, a = 0, nonlinearity = 'relu')
        init.kaiming_uniform_(self.decoder.weight, a = 0, nonlinearity = 'relu')
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        if tied:
            with torch.no_grad():
                self.decoder.weight = nn.Parameter(self.encoder.weight.t().clone())

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.relu(x)
        x = self.decoder(x)
        return x
    
    def sparsity(self, activations):
        mean = torch.mean(activations, dim = 0)
        mean = torch.clamp(mean, self.eps, 1 - self.eps)
        kldiv = (self.sparsity_target * torch.log(self.sparsity_target / mean)) + ((1 - self.sparsity_target) *  torch.log((1 - self.sparsity_target) / (1 - mean)))
        return torch.sum(kldiv)

def test():
    model = NNSAE(10, 3, tied = True)
    print(model.encoder.weight.shape)
    print(model.decoder.weight.shape)
    summary(model, (39996, 10))

if __name__ == "__main__":
    test()