import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        x = x.unsqueeze(0) if x.ndimension() == 1 else x
        assert x.ndimension() == 2 and x.shape[-1] == 3

        freq = (
            2 ** torch.arange(self.levels, device=x.device).unsqueeze(-1)
            * torch.pi
            * x.unsqueeze(1)
        )
        sin_cos = torch.cat([torch.sin(freq), torch.cos(freq)], dim=2)
        return torch.cat([x, sin_cos.flatten(start_dim=1)], -1)
