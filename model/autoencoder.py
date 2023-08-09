import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size = 3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels = 29, hidden_channels = 3):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = hidden_channels,
                      kernel_size = (1, 1),
                      stride = 1,
                      padding = 0),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels = hidden_channels,
                      out_channels = in_channels,
                      kernel_size = (1, 1),
                      stride = 1,
                      padding = 0,)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x