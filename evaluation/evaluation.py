from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device

        model_device = next(model.parameters()).device
        if model_device != self.device:
            self.model.to(self.device)

    def calculate_loss_batch(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        if input_batch.device != self.device:
            input_batch = input_batch.to(self.device)
        if target_batch.device != self.device:
            target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)
        flat_logits = logits.flatten(0, 1)
        flat_targets = target_batch.flatten()

        loss = nn.functional.cross_entropy(flat_logits, flat_targets)

        return loss

    @torch.no_grad()
    def calculate_loss_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> float:
        if len(dataloader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        total_loss = 0.0

        for i, (input_batch, target_batch) in enumerate(dataloader):
            if i < num_batches:
                loss = self.calculate_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch,
                )
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches

    def evaluate_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        iter_evaluation: int,
    ) -> Tuple[float, float]:
        self.model.eval()

        train_loss = self.calculate_loss_dataloader(
            dataloader=train_dataloader,
            num_batches=iter_evaluation,
        )
        val_loss = self.calculate_loss_dataloader(
            dataloader=val_dataloader,
            num_batches=iter_evaluation,
        )

        self.model.train()

        return train_loss, val_loss

    def print_metrics(
        self,
        train_loss: float,
        val_loss: float,
    ) -> None:
        # Calculate Perplexity
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        val_ppl = torch.exp(torch.tensor(val_loss)).item()

        header = f"{'Metric':<12} | {'Train':<10} | {'Validation':<10}"
        sep = "-" * len(header)

        print(sep)
        print(header)
        print(sep)
        print(f"{'Loss':<12} | {train_loss:<10.4f} | {val_loss:<10.4f}")
        print(f"{'Perplexity':<12} | {train_ppl:<10.2f} | {val_ppl:<10.2f}")
        print(sep)
