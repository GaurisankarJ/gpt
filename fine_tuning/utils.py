import os
from datetime import datetime
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from generate import Generator_Qwen_3


def save_csv_logs(
    data: dict,
    name: str,
) -> str:
    folder = "logs"
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_frame = pd.DataFrame(data)
    file_name = f"{name}_{timestamp}.csv"
    full_path = os.path.join(folder, file_name)

    data_frame.to_csv(full_path, index=False)

    return full_path


def generate_prompt(
    generator: Generator_Qwen_3,
    prompt: str,
    max_token_length: int = 50,
) -> str:
    output = generator.generate(
        idx=generator.text_to_token_ids(prompt).unsqueeze(0),
        max_token_length=max_token_length,
        cache_enabled=True,
    )
    print(
        f"Sample Model Output: \n{generator.token_ids_to_text(output.squeeze(0)).strip()}"
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    global_steps: int,
    tokens_seen: int = 0,
    train_losses: list[float] = [],
    val_losses: list[float] = [],
    total_tokens_seen: int = 0,
    epochs_eval: int = 0,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder = "checkpoints"

    os.makedirs(folder, exist_ok=True)

    file_name = f"checkpoint_{model_name}_ep{epoch + 1:02d}_step{global_steps:06d}_instruct_{timestamp}.pth"
    full_path = os.path.join(folder, file_name)

    torch.save(
        {
            "epoch": epoch,
            "global_steps": global_steps,
            "tokens_seen": tokens_seen,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_tokens_seen": total_tokens_seen,
            "epochs_eval": epochs_eval,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        full_path,
    )

    print(f"Checkpoint for {model_name} saved to: {full_path}.")

    return full_path


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
