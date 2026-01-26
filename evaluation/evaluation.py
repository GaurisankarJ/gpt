from typing import Literal, Optional, Tuple

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

    def calculate_loss_batch_classification(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        if input_batch.device != self.device:
            input_batch = input_batch.to(self.device)
        if target_batch.device != self.device:
            target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)
        last_logit = logits[:, -1, :]

        loss = nn.functional.cross_entropy(last_logit, target_batch)

        return loss

    @torch.no_grad()
    def calculate_loss_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
        task: Optional[Literal["pre_training", "classification"]] = "pre_training",
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
                if task == "pre_training":
                    loss = self.calculate_loss_batch(
                        input_batch=input_batch,
                        target_batch=target_batch,
                    )
                elif task == "classification":
                    loss = self.calculate_loss_batch_classification(
                        input_batch=input_batch,
                        target_batch=target_batch,
                    )
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches

    @torch.no_grad()
    def calculate_accuracy_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ):
        self.model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        for i, (input_batch, target_batch) in enumerate(dataloader):
            if i < num_batches:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)
                last_logit = logits[:, -1, :]
                probs = torch.softmax(last_logit, dim=-1)
                predicted_labels = torch.argmax(probs, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

        return correct_predictions / num_examples

    def evaluate_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        iter_evaluation: int,
        task: Optional[Literal["pre_training", "classification"]] = "pre_training",
    ) -> Tuple[float, float, float | None, float | None]:
        self.model.eval()

        train_loss = self.calculate_loss_dataloader(
            dataloader=train_dataloader,
            num_batches=iter_evaluation,
            task=task,
        )
        val_loss = self.calculate_loss_dataloader(
            dataloader=val_dataloader,
            num_batches=iter_evaluation,
            task=task,
        )
        if task == "classification":
            train_accuracy = self.calculate_accuracy_dataloader(
                dataloader=train_dataloader,
                num_batches=iter_evaluation,
            )
            val_accuracy = self.calculate_accuracy_dataloader(
                dataloader=val_dataloader,
                num_batches=iter_evaluation,
            )
        else:
            train_accuracy = None
            val_accuracy = None

        self.model.train()

        return (train_loss, val_loss, train_accuracy, val_accuracy)

    def print_metrics_pre_training(
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

    def print_metrics_classification(
        self,
        train_accuracy: float,
        val_accuracy: float,
        test_accuracy: float,
    ) -> None:
        header = f"{'Metric':<12} | {'Train':<10} | {'Val':<10} | {'Test':<10}"
        sep = "-" * len(header)

        print(f"\n{sep}")
        print(header)
        print(sep)

        print(
            f"{'Accuracy':<12} | "
            f"{train_accuracy * 100:<9.2f}% | "
            f"{val_accuracy * 100:<9.2f}% | "
            f"{test_accuracy * 100:<9.2f}%"
        )
        print(sep)
