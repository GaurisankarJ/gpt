from typing import Optional, Tuple

import torch
from torch import nn

from attention_layer import GroupedQueryAttention_Qwen_3
from feed_forward_layers import FeedForward_Qwen_3
from normalization import RMSNorm


class TransformerBlock_Qwen_3(nn.Module):
    def __init__(
        self,
        dim_embedding,
        num_heads,
        dim_head,
        num_kv_groups,
        qk_norm,
        dim_hidden,
        dtype,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.att = GroupedQueryAttention_Qwen_3(
            d_in=dim_embedding,
            num_heads=num_heads,
            dim_head=dim_head,
            num_kv_groups=num_kv_groups,
            qk_norm=qk_norm,
            dtype=dtype,
        )
        self.ff = FeedForward_Qwen_3(
            dim_embedding=dim_embedding,
            dim_hidden=dim_hidden,
            dtype=dtype,
        )
        self.norm1 = RMSNorm(dim_embedding, eps=eps)
        self.norm2 = RMSNorm(dim_embedding, eps=eps)

    def forward(
        self,
        x,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache
