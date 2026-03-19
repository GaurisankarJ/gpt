import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.utils import (  # noqa: E402
    calculate_average_joint_log_probability,
    calculate_next_token_probabilities,
)


class NextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=idx.device)
        for b in range(batch_size):
            for t in range(seq_len - 1):
                target = int(idx[b, t + 1].item())
                logits[b, t, target] = 10.0
        return logits


def test_calculate_next_token_probabilities_returns_aligned_probs():
    model = NextTokenModel(vocab_size=10)
    idx = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    next_probs, next_log_probs, joint_probs, joint_log_probs = (
        calculate_next_token_probabilities(
            model=model,
            idx=idx,
            device=torch.device("cpu"),
        )
    )

    assert next_probs.shape == (2, 3)
    assert next_log_probs.shape == (2, 3)
    assert joint_probs.shape == (2,)
    assert joint_log_probs.shape == (2,)
    assert torch.all(next_probs > 0.99)
    assert torch.all(joint_probs > 0.99)
    assert torch.allclose(
        next_log_probs,
        torch.log(next_probs.clamp_min(1e-12)),
    )
    assert torch.allclose(
        joint_log_probs,
        next_log_probs.sum(dim=-1),
    )
    assert torch.allclose(
        joint_probs,
        torch.exp(joint_log_probs),
    )


def test_calculate_next_token_probabilities_validates_input_shape():
    model = NextTokenModel(vocab_size=10)
    with pytest.raises(ValueError, match="2D tensor"):
        calculate_next_token_probabilities(
            model=model,
            idx=torch.tensor([1, 2, 3], dtype=torch.long),
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="at least 2 tokens"):
        calculate_next_token_probabilities(
            model=model,
            idx=torch.tensor([[1]], dtype=torch.long),
            device=torch.device("cpu"),
        )


def test_calculate_average_joint_log_probability_matches_suffix_mean():
    model = NextTokenModel(vocab_size=10)
    idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    prompt_idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    _, next_log_probs, _, _ = calculate_next_token_probabilities(
        model=model,
        idx=idx,
        device=torch.device("cpu"),
    )
    expected = next_log_probs[:, prompt_idx.shape[-1] - 1 :].mean().item()
    observed = calculate_average_joint_log_probability(
        model=model,
        prompt_idx=prompt_idx,
        idx=idx,
        device=torch.device("cpu"),
    )

    assert observed == pytest.approx(expected)


def test_calculate_average_joint_log_probability_requires_continuation_tokens():
    model = NextTokenModel(vocab_size=10)
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    prompt_idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="continuation token"):
        calculate_average_joint_log_probability(
            model=model,
            prompt_idx=prompt_idx,
            idx=idx,
            device=torch.device("cpu"),
        )
