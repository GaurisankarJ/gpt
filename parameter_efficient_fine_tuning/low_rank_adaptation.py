import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(
            self.A, a=math.sqrt(5)
        )  # similar to standard weight initialization
        self.B = nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype, device=device))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.A.dtype:
            x = x.to(self.A.dtype)
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            linear.weight.dtype,
            linear.weight.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)
