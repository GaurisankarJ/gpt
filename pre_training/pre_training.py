import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from evaluation import Evaluator
from generate import Generator
from torch.utils.data import DataLoader


class BasicTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.total_tokens_seen = []
        self.epochs_eval = []
        self.tokens_seen = 0
        self.global_steps = -1
        self.evaluator = Evaluator(
            model=model,
            device=device,
        )

    def print_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        tokens_m = self.tokens_seen / 1e6

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Step {self.global_steps:06d} | "
            f"Loss (T/V): {train_loss:.4f} / {val_loss:.4f} | "
            f"Tokens: {tokens_m:.4f}M"
        )

    def save_checkpoint(
        self,
        model_name: str,
        epoch: int,
    ) -> None:
        folder = "checkpoints"

        os.makedirs(folder, exist_ok=True)

        file_name = (
            f"checkpoint_{model_name}_ep{epoch + 1:02d}_step{self.global_steps:06d}.pth"
        )
        full_path = os.path.join(folder, file_name)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            full_path,
        )

        print(f"Checkpoint saved to: {full_path}.")

    def save_csv(
        self,
        model_name: str,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        folder = "logs"

        os.makedirs(folder, exist_ok=True)

        data = {
            "Epochs": self.epochs_eval,
            "Tokens Seen": self.total_tokens_seen,
            "Training Loss": self.train_losses,
            "Validation Loss": self.val_losses,
        }
        metric_dataframe = pd.DataFrame(data)

        file_name = f"{model_name}_{timestamp}.csv"
        full_path = os.path.join(folder, file_name)

        metric_dataframe.to_csv(full_path, index=False)

        print(f"Metrics saved to: {full_path}.")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        freq_evaluation: int = 1000,
        iter_evaluation: Optional[int] = 100,
        freq_checkpoint: Optional[int] = None,
        model_name: Optional[str] = "gpt",
        save_logs: Optional[bool] = False,
        generator: Optional[Generator] = None,
    ) -> Tuple[float, float, int]:
        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in train_dataloader:
                input_batch, target_batch = (
                    input_batch.to(self.device),
                    target_batch.to(self.device),
                )

                self.optimizer.zero_grad()

                loss = self.evaluator.calculate_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch,
                )

                loss.backward()
                self.optimizer.step()

                self.tokens_seen += input_batch.numel()
                self.global_steps += 1

                if (
                    self.global_steps % freq_evaluation == 0
                    or (self.global_steps + 1) % len(train_dataloader) == 0
                ):
                    train_loss, val_loss = self.evaluator.evaluate_model(
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        iter_evaluation=iter_evaluation,
                    )

                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.total_tokens_seen.append(self.tokens_seen)

                    epoch_steps = self.global_steps / len(train_dataloader)
                    self.epochs_eval.append(round(epoch_steps, 3))

                    self.print_metrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

            if freq_checkpoint and (epoch + 1) % freq_checkpoint == 0:
                self.save_checkpoint(model_name=model_name, epoch=epoch)

            if generator:
                output = generator.generate(
                    idx=generator.text_to_token_ids("Hello World!").unsqueeze(0),
                    max_token_length=50,
                )
                print(
                    f"Sample Model Ouput: \n{generator.token_ids_to_text(output.squeeze(0))}"
                )

        if save_logs:
            self.save_csv(
                model_name=model_name,
            )

        self.save_checkpoint(model_name=f"{model_name}_final", epoch=epoch + 1)

        return self.train_losses, self.val_losses, self.total_tokens_seen
