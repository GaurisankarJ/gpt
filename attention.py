import torch
import torch.nn as nn


# Naive Attention Layer
class CausalAttention(nn.Module):
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

    def forward(self, x):
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


class MultiHeadAttentionV1(nn.Module):
    def __init__(self, dim_embedding, num_head, context_length, dropout, qkv_bias):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(
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

    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.dropout(self.project(out))

        return out


# Multi Head Attention Layer (Efficient)
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embedding, num_head, context_length, dropout, qkv_bias):
        super().__init__()

        assert dim_embedding % num_head == 0, (
            "Embedding dimension must be divisible by number of heads."
        )

        self.num_head = num_head
        self.dim_head = dim_embedding // num_head

        self.weights_key = nn.Linear(dim_embedding, dim_embedding, bias=qkv_bias)
        self.weights_query = nn.Linear(dim_embedding, dim_embedding, bias=qkv_bias)
        self.weights_value = nn.Linear(dim_embedding, dim_embedding, bias=qkv_bias)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_length, context_length)), diagonal=0)
        )
        self.project = nn.Linear(dim_embedding, dim_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, channel = x.shape

        keys = self.weights_key(x)
        queries = self.weights_query(x)
        values = self.weights_value(x)

        keys = keys.view(batch, time, self.num_head, self.dim_head).transpose(1, 2)
        queries = queries.view(batch, time, self.num_head, self.dim_head).transpose(
            1, 2
        )
        values = values.view(batch, time, self.num_head, self.dim_head).transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(self.tril[:time, :time] == 0, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores * self.dim_head**-0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch, time, channel)
        context_vector = self.project(context_vector)

        return context_vector
