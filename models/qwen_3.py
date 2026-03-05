from typing import Optional

import torch
import torch.nn as nn

from kv_cache import KVCache
from normalization import RMSNorm
from positional_encoding import compute_rope_params
from transformer_blocks import TransformerBlock_Qwen_3


class Qwen_3_Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_embedding: int,
        num_layers: int,
        num_heads: int,
        dim_head: Optional[int],
        num_kv_groups: int,
        qk_norm: bool,
        dim_hidden: int,
        rope_base: float,
        context_length: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(
            vocab_size,
            dim_embedding,
            dtype=dtype,
        )

        # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock_Qwen_3(
                    dim_embedding=dim_embedding,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    num_kv_groups=num_kv_groups,
                    qk_norm=qk_norm,
                    dim_hidden=dim_hidden,
                    dtype=dtype,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(dim_embedding)
        self.out_head = nn.Linear(
            dim_embedding,
            vocab_size,
            bias=False,
            dtype=dtype,
        )

        # Reusable utilities
        if dim_head is None:
            dim_head = dim_embedding // num_heads
        else:
            dim_head = dim_head
        cos, sin = compute_rope_params(
            dim_head=dim_head,
            theta_base=rope_base,
            context_length=context_length,
            dtype=dtype,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.current_pos = 0  # Track current position in KV cache
        self.dtype = dtype
        self.num_layers = num_layers

    def forward(
        self,
        in_idx: torch.Tensor,
        cache: Optional[KVCache] = None,
    ):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(
                x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache
            )
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))

        return logits

    def reset_kv_cache(self):
        self.current_pos = 0
