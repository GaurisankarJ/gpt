import torch
import torch.nn as nn
from transformer_blocks import TransformerBlock_GPT_2
from utils import LayerNorm

GPT2_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "dim_embedding": 768,  # Embedding dimension
    "num_heads": 12,  # Number of attention heads
    "num_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


class GPT_2_Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        dim_embedding = config["dim_embedding"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        drop_rate = config["drop_rate"]
        qkv_bias = config["qkv_bias"]

        self.token_embedding = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedding = nn.Embedding(context_length, dim_embedding)
        self.drop_embedding = nn.Dropout(drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock_GPT_2(
                    dim_embedding=dim_embedding,
                    num_heads=num_heads,
                    context_length=context_length,
                    dropout=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = LayerNorm(dim_embedding)
        self.out_head = nn.Linear(dim_embedding, vocab_size, bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape

        token_embeddings = self.token_embedding(in_idx)
        position_embeddings = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = token_embeddings + position_embeddings
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.norm_final(x)

        logits = self.out_head(x)

        return logits
