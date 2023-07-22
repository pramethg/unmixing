import torch
import torch.nn as nn

class NNSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_target = 0.1, sparsity_weight = 0.01):
        super(NNSAE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias = False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias = False)
        self.decoder.weight.data = self.encoder.weight.data.t()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.relu(x)
        x = self.decoder(x)
        return x
    
    def sparsity(self, activations):
        mean = torch.mean(activations, dim = 0)
        kldiv = (self.sparsity_target * torch.log(self.sparsity_target / mean)) + ((1 - self.sparsity_target) *  torch.log((1 - self.sparsity_target) / (1 - mean)))
        return torch.sum(kldiv)