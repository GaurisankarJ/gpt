import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForwardV1


# Naive Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim_embedding, num_heads, context_length, dropout, qkv_bias):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            dim_embedding=dim_embedding,
            num_head=num_heads,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.feed_forward = FeedForwardV1(dim_embedding=dim_embedding, dropout=dropout)
        self.norm_one = nn.LayerNorm(dim_embedding)
        self.norm_two = nn.LayerNorm(dim_embedding)
        self.drop_residual = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop_residual(self.multi_head_attention(self.norm_one(x)))
        x = x + self.drop_residual(self.feed_forward(self.norm_two(x)))

        return x
