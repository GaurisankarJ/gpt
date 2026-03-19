from typing import Tuple

import torch
import torch.nn as nn


def calculate_next_token_probabilities(
    model: nn.Module,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if idx.ndim != 2:
        raise ValueError("idx must be a 2D tensor of shape [batch, seq_len].")
    if idx.shape[1] < 2:
        raise ValueError("idx must contain at least 2 tokens per sequence.")

    idx = idx.to(device)
    model.eval()

    logits = model(idx)
    logits = logits[:, :-1, :]

    probs = torch.softmax(logits, dim=-1)

    next_tok_idx = idx[:, 1:]
    next_tok_probs = probs.gather(
        dim=-1,
        index=next_tok_idx.unsqueeze(-1),
    ).squeeze(-1)

    next_tok_log_probs = torch.log(next_tok_probs.clamp_min(1e-12))
    next_tok_joint_log_probs = next_tok_log_probs.sum(dim=-1)
    next_tok_joint_probs = torch.exp(next_tok_joint_log_probs)

    if show:
        print(f"Next token probabilities: {next_tok_probs}")
        print(f"Next token joint probabilities: {next_tok_joint_probs}")
        print(f"Next token log probabilities: {next_tok_log_probs}")
        print(f"Next token joint log probabilities: {next_tok_joint_log_probs}")

    return (
        next_tok_probs,
        next_tok_joint_probs,
        next_tok_log_probs,
        next_tok_joint_log_probs,
    )


def calculate_average_log_probability(
    model: nn.Module,
    prompt_idx: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> float:
    if prompt_idx.ndim == 0:
        raise ValueError("prompt_idx must contain at least one token.")

    prompt_length = prompt_idx.shape[-1]
    if prompt_length < 1:
        raise ValueError("prompt_idx must contain at least one token.")
    if idx.shape[-1] <= prompt_length:
        raise ValueError(
            "idx must contain at least one continuation token beyond prompt_idx."
        )

    _, _, next_tok_log_probs, _ = calculate_next_token_probabilities(
        model=model,
        idx=idx,
        device=device,
        show=show,
    )

    start_idx = prompt_length - 1

    return next_tok_log_probs[:, start_idx:].mean().item()
