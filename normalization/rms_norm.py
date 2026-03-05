import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim_embedding,
        eps: float = 1e-6,
        bias: bool = False,
        qwen3_compatible: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(dim_embedding))
        self.shift = nn.Parameter(torch.zeros(dim_embedding)) if bias else None

    def forward(
        self,
        x,
    ):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
