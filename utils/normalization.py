import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim_embedding):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(dim_embedding))
        self.shift = nn.Parameter(torch.zeros(dim_embedding))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        var = x.var(dim=-1, keepdims=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift
