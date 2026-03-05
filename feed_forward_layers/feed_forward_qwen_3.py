from typing import Optional

import torch.nn as nn


class FeedForward_Qwen_3(nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_hidden,
        dtype,
        bias: Optional[bool] = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_embedding, dim_hidden, dtype=dtype, bias=bias)
        self.fc2 = nn.Linear(dim_embedding, dim_hidden, dtype=dtype, bias=bias)
        self.fc3 = nn.Linear(dim_hidden, dim_embedding, dtype=dtype, bias=bias)

    def forward(
        self,
        x,
    ):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2

        return self.fc3(x)
