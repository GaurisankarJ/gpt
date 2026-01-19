import torch
import torch.nn as nn


# Naive Feed Forward Layer
class GELU(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForwardV1(nn.Module):
    def __init__(self, dim_embedding, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_embedding, 4 * dim_embedding),
            GELU(),
            nn.Linear(4 * dim_embedding, dim_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
