from typing import Any, Optional, Set

import tiktoken
import torch

DATA_PATH = "data/"


# Get MPS if available
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS device")
        device = torch.device("mps")
    else:
        print("MPS device not found, falling back to CPU")
        device = torch.device("cpu")

    return device


# Encode/Decode GPT2
def text_to_token_ids(
    text: str,
    tokenizer: Optional[Any] = None,
    allowed_special: Optional[Set[str]] = {"<|endoftext|>"},
) -> torch.Tensor:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    return torch.tensor(tokenizer.encode(text, allowed_special=allowed_special))


def token_ids_to_text(
    token_ids: torch.Tensor,
    tokenizer: Optional[Any] = None,
) -> str:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer.decode(token_ids.tolist())


# Model
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_gpt2_parameters(model: torch.nn.Module) -> str:
    total_params = count_parameters(model)
    out_head = count_parameters(model.out_head)

    return f"{total_params - out_head:,} Parameters"
