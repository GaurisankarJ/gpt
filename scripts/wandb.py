from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def build_wandb_config(args) -> dict:
    return {
        "model_name": args.model_name,
        "model_size": args.model_size,
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "dataset_file_path": args.dataset_file_path,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "freq_evaluation": args.freq_evaluation,
        "iter_evaluation": args.iter_evaluation,
        "max_length": args.max_length,
        "seed": args.seed,
        "warmup": args.warmup,
        "cosine_decay": args.cosine_decay,
        "initial_learning_rates": args.initial_learning_rates,
        "peak_learning_rates": args.peak_learning_rates,
        "learning_rate_warmup_percentage": args.learning_rate_warmup_percentage,
        "minimum_learning_rates_percentage": args.minimum_learning_rates_percentage,
        "lora": args.lora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "wandb_artifacts": args.wandb_artifacts,
        "train": args.train,
        "test": args.test,
        "eval": args.eval,
    }


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total_params, trainable_params


class WandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: Optional[str],
        run_name: Optional[str],
        entity: Optional[str],
        tags: Optional[list[str]],
        config: dict,
    ) -> None:
        self.enabled = enabled
        self.wandb: Any = None
        self.run: Any = None

        if not self.enabled:
            return

        try:
            import wandb as wandb_module  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "wandb is enabled but not installed. Install with `pip install wandb` "
                "or run with `--no-wandb`."
            ) from exc

        self.wandb = wandb_module
        self.run = self.wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            tags=tags,
            config=config,
        )
        self.define_default_metrics()

    def define_default_metrics(self) -> None:
        if self.run is None:
            return
        self.wandb.define_metric("train/global_step")
        self.wandb.define_metric("train/*", step_metric="train/global_step")
        self.wandb.define_metric("eval/global_step")
        self.wandb.define_metric("eval/*", step_metric="eval/global_step")
        self.wandb.define_metric("epoch/index")
        self.wandb.define_metric("epoch/*", step_metric="epoch/index")

    def log(self, payload: dict) -> None:
        if self.run is None:
            return
        self.run.log(payload)

    def set_summary(self, key: str, value: Any) -> None:
        if self.run is None:
            return
        self.run.summary[key] = value

    def watch_model(self, model: torch.nn.Module, log_freq: int) -> None:
        if self.run is None or not hasattr(self.wandb, "watch"):
            return
        self.wandb.watch(model, log="all", log_freq=log_freq)

    def log_dataset_artifact(
        self,
        dataset_file_path: str,
        model_name: str,
        seed: int,
        shuffle_before_split: bool,
    ) -> None:
        if self.run is None or not hasattr(self.wandb, "Artifact"):
            return
        dataset_path = Path(dataset_file_path)
        if not dataset_path.exists():
            return
        dataset_artifact = self.wandb.Artifact(
            name=f"{model_name}-dataset",
            type="dataset",
            metadata={
                "dataset_file_path": str(dataset_path),
                "seed": seed,
                "shuffle_before_split": shuffle_before_split,
            },
        )
        dataset_artifact.add_file(str(dataset_path))
        self.run.log_artifact(dataset_artifact)

    def log_checkpoint(
        self,
        model_name: str,
        checkpoint_path: str,
        checkpoint_metrics: dict,
    ) -> None:
        if self.run is None:
            return
        self.run.log(checkpoint_metrics)
        if not hasattr(self.wandb, "Artifact"):
            return
        checkpoint_artifact = self.wandb.Artifact(
            name=f"{model_name}-checkpoint-ep{checkpoint_metrics['checkpoint/epoch']}",
            type="model",
            metadata=checkpoint_metrics,
        )
        checkpoint_artifact.add_file(checkpoint_path)
        self.run.log_artifact(checkpoint_artifact)

    def log_response_table(self, test_data: list[dict], max_rows: int = 20) -> None:
        if self.run is None or not test_data or not hasattr(self.wandb, "Table"):
            return
        response_table = self.wandb.Table(
            columns=["instruction", "input", "output", "model_response"]
        )
        for row in test_data[: min(max_rows, len(test_data))]:
            response_table.add_data(
                row.get("instruction", ""),
                row.get("input", ""),
                row.get("output", ""),
                row.get("model_response", ""),
            )
        self.run.log({"test/generated_samples": response_table})

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
