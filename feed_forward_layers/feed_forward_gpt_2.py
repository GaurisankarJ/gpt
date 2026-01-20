import torch.nn as nn

from .activation import GELU


class FeedForward_GPT_2(nn.Module):
    def __init__(
        self,
        dim_embedding,
        dropout,
        expansion_factor=4,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_embedding, expansion_factor * dim_embedding),
            GELU(),
            nn.Linear(expansion_factor * dim_embedding, dim_embedding),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
    ):
        return self.layers(x)
