import math
import os
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation import EvaluatorInstructionFineTuning
from generate import Generator_Qwen_3


class TrainerInstructionFineTuning:
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
        self.track_learning_rate = []
        self.warmup_steps = None
        self.initial_learning_rates = None
        self.peak_learning_rates = None
        self.evaluator = EvaluatorInstructionFineTuning(
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

    def print_epoch_footer(
        self,
        epoch: int,
        epoch_avg_loss: float,
        epoch_time_seconds: float,
    ) -> None:
        tokens_m = self.tokens_seen / 1e6
        tokens_per_second = self.tokens_seen / max(
            time.time() - self.training_start_time, 1e-9
        )

        print(
            f"Epoch {epoch + 1:02d} done | "
            f"Avg Loss: {epoch_avg_loss:.4f} | "
            f"Epoch Time: {epoch_time_seconds / 60:.2f}m | "
            f"Tokens: {tokens_m:.4f}M | "
            f"Throughput: {tokens_per_second:.0f} tok/s"
        )

    def save_checkpoint(
        self,
        model_name: str,
        epoch: int,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        folder = "checkpoints"

        os.makedirs(folder, exist_ok=True)

        file_name = f"checkpoint_{model_name}_ep{epoch + 1:02d}_step{self.global_steps:06d}_instruct_{timestamp}.pth"
        full_path = os.path.join(folder, file_name)

        torch.save(
            {
                "epoch": epoch,
                "global_steps": self.global_steps,
                "tokens_seen": self.tokens_seen,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "total_tokens_seen": self.total_tokens_seen,
                "epochs_eval": self.epochs_eval,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            full_path,
        )

        print(f"Checkpoint for instruction fine-tuning saved to: {full_path}.")

    def save_csv(
        self,
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

    def save_csv_metrics(
        self,
        model_name: str,
    ) -> None:
        data = {
            "Epochs": self.epochs_eval,
            "Total Tokens Seen": self.total_tokens_seen,
            "Training Loss": self.train_losses,
            "Validation Loss": self.val_losses,
        }

        full_path = self.save_csv(
            data=data,
            name=f"{model_name}_instruct_metrics",
        )

        print(f"Metrics for instruction fine-tuning saved to: {full_path}.")

    def save_csv_learning_rate(
        self,
        model_name: str,
    ) -> None:
        data = {
            "Global Steps": range(len(self.track_learning_rate)),
            "Learning Rate": self.track_learning_rate,
        }

        full_path = self.save_csv(
            data=data,
            name=f"{model_name}_instruct_learning_rate",
        )

        print(f"Learning rates for instruction fine-tuning saved to: {full_path}.")

    def reset_training_state(self) -> None:
        self.train_losses.clear()
        self.val_losses.clear()
        self.total_tokens_seen.clear()
        self.epochs_eval.clear()
        self.tokens_seen = 0
        self.global_steps = -1
        self.track_learning_rate.clear()
        self.warmup_steps = None
        self.learning_rate_increment = None
        self.initial_learning_rates = None
        self.peak_learning_rates = None

    def set_learning_rate_warmup(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        initial_learning_rate: Optional[float] = None,
        learning_rate_warmup: Optional[float] = None,
    ) -> None:
        if learning_rate_warmup is None:
            return
        if learning_rate_warmup < 0 or learning_rate_warmup > 100:
            raise ValueError("Learning rate warmup must be between 0 and 100.")

        total_steps = len(train_dataloader) * num_epochs
        if total_steps <= 0:
            raise ValueError("Training requires at least one optimization step.")

        warmup_steps = int(learning_rate_warmup * total_steps / 100)
        if learning_rate_warmup > 0:
            warmup_steps = max(1, warmup_steps)
        else:
            warmup_steps = 0

        if warmup_steps == 0:
            self.warmup_steps = None
            self.learning_rate_increment = None
            self.initial_learning_rates = None
            self.peak_learning_rates = None

            return

        peak_learning_rates = [
            float(param_group["lr"]) for param_group in self.optimizer.param_groups
        ]
        if initial_learning_rate is None:
            initial_learning_rates = [lr / 1e4 for lr in peak_learning_rates]
        else:
            if initial_learning_rate < 0:
                raise ValueError("Initial learning rate must be non-negative.")
            initial_learning_rates = [
                float(initial_learning_rate) for _ in self.optimizer.param_groups
            ]

        learning_rate_increment = [
            (peak - init) / max(warmup_steps - 1, 1)
            for init, peak in zip(initial_learning_rates, peak_learning_rates)
        ]

        self.warmup_steps = warmup_steps
        self.learning_rate_increment = learning_rate_increment
        self.initial_learning_rates = initial_learning_rates
        self.peak_learning_rates = peak_learning_rates

        for param_group, group_initial_lr in zip(
            self.optimizer.param_groups, initial_learning_rates
        ):
            param_group["lr"] = group_initial_lr

    def learning_rate_schedule(
        self,
        next_step: int,
        cosine_decay: bool,
        num_epochs: int,
        train_dataloader: DataLoader,
    ) -> None:
        if next_step < self.warmup_steps:
            if self.warmup_steps == 1:
                progress = 1.0
            else:
                progress = next_step / (self.warmup_steps - 1)
            for idx, param_group in enumerate(self.optimizer.param_groups):
                initial_lr = self.initial_learning_rates[idx]
                peak_lr = self.peak_learning_rates[idx]
                param_group["lr"] = initial_lr + progress * (peak_lr - initial_lr)
        else:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.peak_learning_rates[idx]

            if cosine_decay and self.global_steps > self.warmup_steps:
                progress = (self.global_steps - self.warmup_steps) / (
                    num_epochs * len(train_dataloader) - self.warmup_steps
                )
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    initial_lr = self.initial_learning_rates[idx]
                    peak_lr = self.peak_learning_rates[idx]
                    param_group["lr"] = initial_lr + 0.5 * (peak_lr - initial_lr) * (
                        1 + math.cos(math.pi * progress)
                    )

        if self.warmup_steps is None and cosine_decay:
            progress = self.global_steps / (num_epochs * len(train_dataloader))
            for idx, param_group in enumerate(self.optimizer.param_groups):
                initial_lr = self.initial_learning_rates[idx]
                peak_lr = self.peak_learning_rates[idx]
                param_group["lr"] = initial_lr + 0.5 * (peak_lr - initial_lr) * (
                    1 + math.cos(math.pi * progress)
                )

        self.track_learning_rate.append(
            [param_group["lr"] for param_group in self.optimizer.param_groups]
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        freq_evaluation: int = 50,
        iter_evaluation: Optional[int] = 50,
        freq_checkpoint: Optional[int] = None,
        model_name: Optional[str] = "gpt",
        save_logs: Optional[bool] = False,
        generator: Optional[Generator_Qwen_3 | Any] = None,
        show_progress_bar: Optional[bool] = True,
        progress_update_freq: Optional[int] = 20,
        grad_clip: Optional[float] = None,
        initial_learning_rate: Optional[float] = None,
        learning_rate_warmup: Optional[float] = None,  # in percentage
        cosine_decay: Optional[bool] = False,
    ) -> Tuple[List[float], List[float], List[int]]:
        self.reset_training_state()
        self.training_start_time = time.time()

        if learning_rate_warmup is not None:
            self.set_learning_rate_warmup(
                train_dataloader=train_dataloader,
                num_epochs=num_epochs,
                initial_learning_rate=initial_learning_rate,
                learning_rate_warmup=learning_rate_warmup,
            )

        for epoch in range(num_epochs):
            self.model.train()

            epoch_start_time = time.time()
            epoch_running_loss = 0.0
            epoch_steps = 0
            epoch_start_tokens = self.tokens_seen

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                total=len(train_dataloader),
                leave=True,
                disable=not show_progress_bar,
                dynamic_ncols=True,
            )

            for input_batch, target_batch in progress_bar:
                input_batch, target_batch = (
                    input_batch.to(self.device),
                    target_batch.to(self.device),
                )

                self.optimizer.zero_grad(set_to_none=True)

                loss = self.evaluator.calculate_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch,
                )

                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip
                    )

                next_step = self.global_steps + 1
                if self.warmup_steps is not None or cosine_decay:
                    self.learning_rate_schedule(
                        next_step=next_step,
                        cosine_decay=cosine_decay,
                        num_epochs=num_epochs,
                        train_dataloader=train_dataloader,
                    )

                self.optimizer.step()

                step_loss = float(loss.item())
                epoch_running_loss += step_loss
                epoch_steps += 1
                self.tokens_seen += (target_batch != -100).sum().item()
                self.global_steps += 1

                if epoch_steps % progress_update_freq == 0 or epoch_steps == len(
                    train_dataloader
                ):
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed_epoch_seconds = time.time() - epoch_start_time
                    epoch_tokens_processed = self.tokens_seen - epoch_start_tokens
                    tokens_per_second = epoch_tokens_processed / max(
                        elapsed_epoch_seconds, 1e-9
                    )
                    avg_epoch_loss = epoch_running_loss / max(epoch_steps, 1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_epoch_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "tok/s": f"{tokens_per_second:.0f}",
                            "step": f"{self.global_steps}",
                        }
                    )

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

                    epoch_progress = self.global_steps / len(train_dataloader)
                    self.epochs_eval.append(epoch_progress)

                    self.print_metrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

                    progress_bar.set_postfix(
                        {
                            "loss": f"{step_loss:.4f}",
                            "train": f"{train_loss:.4f}",
                            "val": f"{val_loss:.4f}",
                            "step": f"{self.global_steps}",
                        }
                    )

            progress_bar.close()

            if freq_checkpoint and (epoch + 1) % freq_checkpoint == 0:
                self.save_checkpoint(model_name=model_name, epoch=epoch)

            if generator:
                prompt = "Can you explain the following sentence: 'The sky is blue.'"

                output = generator.generate(
                    idx=generator.text_to_token_ids(prompt).unsqueeze(0),
                    max_token_length=50,
                    cache_enabled=True,
                )
                print(
                    f"Sample Model Output: \n{generator.token_ids_to_text(output.squeeze(0))[len(prompt) :].strip()}"
                )

            epoch_end_time = time.time()
            epoch_execution_time_seconds = epoch_end_time - epoch_start_time
            avg_epoch_loss = epoch_running_loss / max(epoch_steps, 1)
            self.print_epoch_footer(
                epoch=epoch,
                epoch_avg_loss=avg_epoch_loss,
                epoch_time_seconds=epoch_execution_time_seconds,
            )

            if save_logs:
                self.save_csv_metrics(
                    model_name=model_name,
                )
                self.save_csv_learning_rate(
                    model_name=model_name,
                )

        training_end_time = time.time()
        training_execution_time_minutes = (
            training_end_time - self.training_start_time
        ) / 60
        training_tokens_processed = self.tokens_seen
        training_tokens_per_second = training_tokens_processed / max(
            training_end_time - self.training_start_time, 1e-9
        )
        print(
            f"Training completed in {training_execution_time_minutes:.2f} minutes | "
            f"Tokens Processed: {training_tokens_processed:,} | "
            f"Avg Throughput: {training_tokens_per_second:.0f} tok/s"
        )

        self.save_checkpoint(model_name=f"{model_name}_final", epoch=epoch)

        return self.train_losses, self.val_losses, self.total_tokens_seen
