import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 4
block_size = 8
n_embed = 32
n_head = 4
n_layers = 2

max_iterations = 1000
eval_interval = 100
eval_iterations = 50
learning_rate = 1e-3
per_dropout = 0.2

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    print("MPS device not found, falling back to CPU")
    device = torch.device("cpu")

with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = "".join(sorted(list(set(text))))
vocab_size = len(vocab)

# Lookup Table
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}

# Enocoder/Decoder
encode = lambda x: [stoi[i] for i in x]  # noqa: E731
decode = lambda x: "".join([itos[i] for i in x])  # noqa: E731

data = torch.tensor(encode(text), dtype=torch.long, device=device)

split = int(0.9 * len(data))
x_train = data[:split]
x_val = data[split:]


def get_batch(split):
    data = x_train if split == "train" else x_val
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, dtype=torch.float, bias=False)
        self.query = nn.Linear(n_embed, head_size, dtype=torch.float, bias=False)
        self.value = nn.Linear(n_embed, head_size, dtype=torch.float, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones((block_size, block_size), dtype=torch.float))
        )
        self.dropout = nn.Dropout(per_dropout)

    def forward(self, x):
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = k @ q.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        sa = torch.softmax(wei, dim=-1)
        sa = self.dropout(sa)

        out = sa @ v

        return out


class MultiHeadAttentionV0(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size=head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed, dtype=torch.float)
        self.dropout = nn.Dropout(per_dropout)

    def forward(self, x):
        out = torch.cat([H(x) for H in self.heads], axis=-1)
        out = self.proj(x)
        out = self.dropout(x)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()

        assert n_embed % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads."
        )

        self.num_heads = num_heads
        self.h_embed = n_embed // num_heads

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.projection = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(per_dropout)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, self.num_heads, self.h_embed).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.h_embed).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.h_embed).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores.masked_fill_(self.tril[:T, :T] == 0, -torch.inf)
        attn_weights = torch.softmax(attn_scores * k.shape[-1] ** -0.5, dim=-1)

        out = (attn_weights @ v).transpose(1, 2)
        out = out.contiguous().view(B, T, C)
        out = self.dropout(self.projection(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed, dtype=torch.float),
            nn.Dropout(per_dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()

        self.sa = MultiHeadAttention(n_embed=n_embed, num_heads=num_heads)
        self.ffwd = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed=n_embed, num_heads=n_head) for _ in range(n_layers)]
        )
        self.lnf = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token = self.token_embedding_table(idx)
        pos = self.pos_embedding_table(torch.arange(T, dtype=torch.long, device=device))

        logits = token + pos
        logits = self.blocks(logits)
        logits = self.lnf(logits)
        logits = self.lm_head(logits)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_token_length):
        for _ in range(max_token_length):
            logits, _ = self(idx[:, -batch_size:])
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], axis=1)

        return idx


model = GPT()
print("Model Parameters:", sum([p.numel() for p in model.parameters()]))
model.to(device)


@torch.no_grad()
def estimate_loss():
    loss_final = {}
    split = ["train", "val"]

    model.eval()
    for s in split:
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            x, y = get_batch(s)
            _, loss = model(x, y)
            losses[i] = loss.item()
        loss_final[s] = losses.mean()
    model.train()

    return loss_final


def train():
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iterations):
        x, y = get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            eval_loss = estimate_loss()
            print(
                f"Train Loss: {eval_loss['train']} Validation Loss: {eval_loss['val']}"
            )


if __name__ == "__main__":
    train()

    model.eval()
    text_input = torch.tensor(
        encode("Hello"), dtype=torch.long, device=device
    ).unsqueeze(0)
    output = model.generate(text_input, max_token_length=200).to("cpu")[0]
    print(decode(output.tolist()))
