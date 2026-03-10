import time
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation import EvaluatorInstructionFineTuning
from generate import Generator_Qwen_3

from .scheduler import LearningRateScheduler
from .utils import generate_prompt, save_checkpoint, save_csv_logs


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

        self.evaluator = EvaluatorInstructionFineTuning(
            model=model,
            device=device,
        )

    def reset_training_state(self) -> None:
        self.train_losses.clear()
        self.val_losses.clear()
        self.total_tokens_seen.clear()
        self.epochs_eval.clear()
        self.tokens_seen = 0
        self.global_steps = -1

    def validate_training_configuration(
        self,
        num_epochs: int,
        freq_evaluation: int,
        progress_update_freq: int,
        freq_checkpoint: Optional[int],
        learning_rate_scheduler: Optional[LearningRateScheduler],
        warmup: bool,
        cosine_decay: bool,
        initial_learning_rates: Optional[List[float]],
        peak_learning_rates: Optional[List[float]],
    ) -> None:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be greater than 0.")
        if freq_evaluation <= 0:
            raise ValueError("freq_evaluation must be greater than 0.")
        if progress_update_freq <= 0:
            raise ValueError("progress_update_freq must be greater than 0.")
        if freq_checkpoint is not None and freq_checkpoint <= 0:
            raise ValueError("freq_checkpoint must be greater than 0 when provided.")

        if learning_rate_scheduler is None:
            if warmup or cosine_decay:
                raise ValueError(
                    "learning_rate_scheduler must be provided when warmup or cosine_decay is enabled."
                )
            return

        if warmup or cosine_decay:
            if initial_learning_rates is None or peak_learning_rates is None:
                raise ValueError(
                    "initial_learning_rates and peak_learning_rates are required "
                    "when warmup or cosine_decay is enabled."
                )
            if len(initial_learning_rates) != len(self.optimizer.param_groups):
                raise ValueError(
                    "initial_learning_rates length must match optimizer param groups."
                )
            if len(peak_learning_rates) != len(self.optimizer.param_groups):
                raise ValueError(
                    "peak_learning_rates length must match optimizer param groups."
                )

    def update_learning_rate(
        self,
        warmup: bool,
        cosine_decay: bool,
        global_steps: int,
        learning_rate_scheduler: LearningRateScheduler,
    ) -> None:
        learning_rates = learning_rate_scheduler.get_scheduled_learning_rates(
            warmup=warmup,
            cosine_decay=cosine_decay,
            global_steps=global_steps,
        )
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise ValueError(
                "Scheduled learning rate count must match optimizer param groups."
            )

        for index, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = learning_rates[index]

    def update_progress_bar(
        self,
        progress_bar: tqdm,
        epoch_start_time: float,
        epoch_start_tokens: int,
        epoch_running_loss: float,
        epoch_steps: int,
    ) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        elapsed_epoch_seconds = time.time() - epoch_start_time
        epoch_tokens_processed = self.tokens_seen - epoch_start_tokens
        tokens_per_second = epoch_tokens_processed / max(elapsed_epoch_seconds, 1e-9)
        avg_epoch_loss = epoch_running_loss / max(epoch_steps, 1)
        progress_bar.set_postfix(
            {
                "loss": f"{avg_epoch_loss:.4f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tokens_per_second:.0f}",
                "step": f"{self.global_steps}",
            }
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

    def evaluate_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        iter_evaluation: int,
        epoch: int,
        progress_bar: tqdm,
        step_loss: float,
    ) -> None:
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

    def save_checkpoint(
        self,
        model_name: str,
        epoch: int,
    ) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            model_name=model_name,
            epoch=epoch,
            global_steps=self.global_steps,
            tokens_seen=self.tokens_seen,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            total_tokens_seen=self.total_tokens_seen,
            epochs_eval=self.epochs_eval,
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

    def save_csv_logs_metrics(
        self,
        model_name: str,
    ) -> None:
        data = {
            "Epochs": self.epochs_eval,
            "Total Tokens Seen": self.total_tokens_seen,
            "Training Loss": self.train_losses,
            "Validation Loss": self.val_losses,
        }

        full_path = save_csv_logs(
            data=data,
            name=f"{model_name}_instruct_metrics",
        )

        print(f"Metrics for instruction fine-tuning saved to: {full_path}.")

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
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        warmup: Optional[bool] = False,
        cosine_decay: Optional[bool] = False,
    ) -> Tuple[List[float], List[float], List[int]]:
        self.reset_training_state()
        self.training_start_time = time.time()

        # Validate training configuration
        self.validate_training_configuration(
            num_epochs=num_epochs,
            freq_evaluation=freq_evaluation,
            progress_update_freq=progress_update_freq,
            freq_checkpoint=freq_checkpoint,
            learning_rate_scheduler=learning_rate_scheduler,
            warmup=warmup,
            cosine_decay=cosine_decay,
            initial_learning_rates=learning_rate_scheduler.initial_learning_rates,
            peak_learning_rates=learning_rate_scheduler.peak_learning_rates,
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
                if learning_rate_scheduler is not None:
                    self.update_learning_rate(
                        warmup=warmup,
                        cosine_decay=cosine_decay,
                        global_steps=next_step,
                        learning_rate_scheduler=learning_rate_scheduler,
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
                    self.update_progress_bar(
                        progress_bar=progress_bar,
                        epoch_start_time=epoch_start_time,
                        epoch_start_tokens=epoch_start_tokens,
                        epoch_running_loss=epoch_running_loss,
                        epoch_steps=epoch_steps,
                    )

                if (
                    self.global_steps % freq_evaluation == 0
                    or (self.global_steps + 1) % len(train_dataloader) == 0
                ):
                    self.evaluate_model(
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        iter_evaluation=iter_evaluation,
                        epoch=epoch,
                        progress_bar=progress_bar,
                        step_loss=step_loss,
                    )

            progress_bar.close()

            if freq_checkpoint and (epoch + 1) % freq_checkpoint == 0:
                self.save_checkpoint(model_name=model_name, epoch=epoch)

            if generator:
                generate_prompt(
                    generator=generator,
                    prompt="Can you explain the following sentence: 'The sky is blue.'",
                    max_token_length=50,
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
                self.save_csv_logs_metrics(
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

        self.save_checkpoint(model_name=f"{model_name}_final", epoch=num_epochs - 1)

        return self.train_losses, self.val_losses, self.total_tokens_seen
