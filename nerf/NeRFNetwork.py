import torch
import torch.nn as nn
import torch.nn.functional as F

from .PositionalEncoder import PositionalEncoder


class NeRFNetwork(nn.Module):
    def __init__(self, filter_size=128, encode_levels=6):
        super().__init__()

        encoded_dim = 2 * 3 * encode_levels + 3
        self.encoder = PositionalEncoder(encode_levels)
        self.layers1 = nn.ModuleList(
            [
                nn.Linear(encoded_dim, filter_size),
                nn.Linear(filter_size, filter_size),
                nn.Linear(filter_size, filter_size),
                nn.Linear(filter_size, filter_size),
            ]
        )
        self.layers2 = nn.ModuleList(
            [
                nn.Linear(filter_size + encoded_dim, filter_size),
                nn.Linear(filter_size, filter_size),
                nn.Linear(filter_size, filter_size),
            ]
        )
        self.layers3 = nn.ModuleList(
            [nn.Linear(filter_size + encoded_dim, filter_size)]
        )
        self.output = nn.Linear(filter_size, 4)

    def forward(self, x):
        x = x_encoded = self.encoder(x)
        for layer in self.layers1:
            x = F.relu(layer(x))

        x = torch.cat([x, x_encoded], dim=-1)
        for layer in self.layers2:
            x = F.relu(layer(x))

        x = torch.cat([x, x_encoded], dim=-1)
        for layer in self.layers3:
            x = F.relu(layer(x))

        x = self.output(x)
        x[..., :3] = F.sigmoid(x[..., :3])
        x[..., -1] = F.relu(x[..., -1])

        return x
