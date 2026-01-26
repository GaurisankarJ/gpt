import os
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation import Evaluator


class TrainerClassification:
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
        self.train_accuracy = []
        self.val_accuracy = []
        self.examples_seen = 0
        self.total_examples_seen = []
        self.epochs_eval = []
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
        train_accuracy: float,
        val_accuracy: float,
    ) -> None:
        examples_t = self.examples_seen / 1e3

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Step {self.global_steps:06d} | "
            f"Loss (T/V): {train_loss:.4f} / {val_loss:.4f} | "
            f"Accuracy (T/V): {train_accuracy:.4f} / {val_accuracy:.4f} | "
            f"Examples: {examples_t:.4f}T"
        )

    def save_checkpoint(
        self,
        model_name: str,
        epoch: int,
    ) -> None:
        folder = "checkpoints"

        os.makedirs(folder, exist_ok=True)

        file_name = f"checkpoint_classification_fine_tuned_{model_name}_ep{epoch + 1:02d}_step{self.global_steps:06d}.pth"
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
            "Examples Seen": self.total_examples_seen,
            "Training Loss": self.train_losses,
            "Validation Loss": self.val_losses,
            "Training Accuracy": self.train_accuracy,
            "Validation Accuracy": self.val_accuracy,
        }
        metric_dataframe = pd.DataFrame(data)

        file_name = f"{model_name}_classification_fine_tuned_{timestamp}.csv"
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
    ) -> Tuple[float, float, int]:
        training_start_time = time.time()
        for epoch in range(num_epochs):
            self.model.train()

            epoch_start_time = time.time()
            for input_batch, target_batch in train_dataloader:
                input_batch, target_batch = (
                    input_batch.to(self.device),
                    target_batch.to(self.device),
                )

                self.optimizer.zero_grad()

                loss = self.evaluator.calculate_loss_batch_classification(
                    input_batch=input_batch,
                    target_batch=target_batch,
                )

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                self.examples_seen += input_batch.shape[0]
                self.global_steps += 1

                if (
                    self.global_steps % freq_evaluation == 0
                    or (self.global_steps + 1) % len(train_dataloader) == 0
                ):
                    train_loss, val_loss, train_accuracy, val_accuracy = (
                        self.evaluator.evaluate_model(
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            iter_evaluation=iter_evaluation,
                            task="classification",
                        )
                    )

                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.train_accuracy.append(train_accuracy)
                    self.val_accuracy.append(val_accuracy)
                    self.total_examples_seen.append(self.examples_seen)

                    epoch_steps = self.global_steps / len(train_dataloader)
                    self.epochs_eval.append(round(epoch_steps, 3))

                    self.print_metrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_accuracy=train_accuracy,
                        val_accuracy=val_accuracy,
                    )

            if freq_checkpoint and (epoch + 1) % freq_checkpoint == 0:
                self.save_checkpoint(model_name=model_name, epoch=epoch)

            epoch_end_time = time.time()
            epoch_execution_time_minutes = (epoch_end_time - epoch_start_time) / 60
            print(f"Epoch completed in {epoch_execution_time_minutes:.2f} minutes.")

        training_end_time = time.time()
        training_execution_time_minutes = (training_end_time - training_start_time) / 60
        print(f"Training completed in {training_execution_time_minutes:.2f} minutes.")

        if save_logs:
            self.save_csv(
                model_name=model_name,
            )

        self.save_checkpoint(
            model_name=f"{model_name}_final_classification_fine_tuned", epoch=epoch
        )

        return self.train_losses, self.val_losses, self.total_examples_seen
