import os

import requests
import torch

DATA_PATH = "data/"


# Data
def download_data(url, filename):
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, filename)

    response = requests.get(url, timeout=30)

    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


def get_tiny_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tiny_shakespeare.txt"
    download_data(url, filename)


def get_the_verdict_data():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    filename = "the_verdict.txt"
    download_data(url, filename)


# Model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdims=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx


def count_gpt2_parameters(model):
    total_params = count_parameters(model)
    out_head = count_parameters(model.out_head)

    return f"{total_params - out_head:,} Parameters"
