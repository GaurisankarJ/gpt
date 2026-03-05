from typing import Optional, Tuple

import torch
import torch.nn as nn

from normalization import RMSNorm
from positional_encoding import apply_rope


class GroupedQueryAttention_Qwen_3(nn.Module):
    def __init__(
        self,
        d_in,
        num_heads,
        dim_head,
        num_kv_groups,
        qk_norm,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if dim_head is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `dim_head` is not set"
            )
            dim_head = d_in // num_heads

        self.dim_head = dim_head
        self.d_out = num_heads * dim_head

        self.W_query = nn.Linear(
            d_in,
            self.d_out,
            bias=False,
            dtype=dtype,
        )
        self.W_key = nn.Linear(
            d_in,
            num_kv_groups * dim_head,
            bias=False,
            dtype=dtype,
        )
        self.W_value = nn.Linear(
            d_in,
            num_kv_groups * dim_head,
            bias=False,
            dtype=dtype,
        )

        self.out_proj = nn.Linear(
            self.d_out,
            d_in,
            bias=False,
            dtype=dtype,
        )

        if qk_norm:
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(
            b,
            num_tokens,
            self.num_heads,
            self.dim_head,
        ).transpose(1, 2)
        keys_new = keys.view(
            b,
            num_tokens,
            self.num_kv_groups,
            self.dim_head,
        ).transpose(1, 2)
        values_new = values.view(
            b,
            num_tokens,
            self.num_kv_groups,
            self.dim_head,
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys, values)
        else:
            keys, values = keys_new, values_new
            next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.dim_head**0.5, dim=-1)

        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )

        return self.out_proj(context), next_cache
