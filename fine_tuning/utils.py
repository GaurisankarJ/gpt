from typing import Any

import torch
import torch.nn as nn


def classify_review(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int = None,
    pad_token_id: int = 50256,
):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.position_embedding.weight.shape[0]

    input_ids = input_ids[: min(max_length, supported_context_length)]

    max_length = (
        min(max_length, supported_context_length)
        if max_length
        else supported_context_length
    )
    input_ids = input_ids[:max_length]

    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "Spam" if predicted_label == 1 else "Not Spam"
