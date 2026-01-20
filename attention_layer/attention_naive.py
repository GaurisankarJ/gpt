import torch
import torch.nn as nn


class CausalAttentionHead(nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_head,
        context_length,
        dropout,
        kvq_bias=False,
    ):
        super().__init__()

        self.dim_head = dim_head
        self.weights_key = nn.Linear(dim_embedding, dim_head, bias=kvq_bias)
        self.weights_query = nn.Linear(dim_embedding, dim_head, bias=kvq_bias)
        self.weights_value = nn.Linear(dim_embedding, dim_head, bias=kvq_bias)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_length, context_length)), diagonal=0)
        )
        self.drop_attention = nn.Dropout(dropout)

    def forward(
        self,
        x,
    ):
        _, T, _ = x.shape

        keys = self.weights_key(x)
        queries = self.weights_query(x)
        values = self.weights_value(x)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(self.tril[:T, :T] == 0, -torch.inf)
        attention_weights = torch.softmax(
            attention_scores * self.dim_head**-0.5, dim=-1
        )
        attention_weights = self.drop_attention(attention_weights)

        context_vector = attention_weights @ values

        return context_vector


class MultiHeadAttentionNaive(nn.Module):
    def __init__(
        self,
        dim_embedding,
        num_head,
        context_length,
        dropout,
        qkv_bias,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttentionHead(
                    dim_embedding=dim_embedding,
                    dim_head=dim_embedding // num_head,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_head)
            ]
        )
        self.project = nn.Linear(dim_embedding, dim_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
    ):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.dropout(self.project(out))

        return out
