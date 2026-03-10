from pathlib import Path
import sys
import time

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.instruction import TrainerInstructionFineTuning  # noqa: E402
from fine_tuning.scheduler import LearningRateScheduler  # noqa: E402
import fine_tuning.instruction as instruction_module  # noqa: E402


class TinyModel(nn.Module):
    """Small model with exactly 20 trainable parameters."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 2)  # 10 * 2 = 20 params

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class TinyNumberGenerator:
    def generate_numbers(self, n: int) -> list[int]:
        return list(range(1, n + 1))


class FakeProgressBar:
    def __init__(self):
        self.last_postfix = None

    def set_postfix(self, values):
        self.last_postfix = values


class FakeEvaluatorInstructionFineTuning:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.evaluate_calls = 0

    def calculate_loss_batch(self, input_batch, target_batch):
        del target_batch
        output = self.model(input_batch)
        return (output**2).mean()

    def evaluate_model(self, train_dataloader, val_dataloader, iter_evaluation):
        del train_dataloader, val_dataloader, iter_evaluation
        self.evaluate_calls += 1
        base = 0.25 + 0.05 * self.evaluate_calls
        return base, base + 0.1


def make_dataloader(batch_size: int = 2) -> DataLoader:
    inputs = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [1, 2, 3, -100],
            [2, 3, 4, -100],
            [3, 4, 5, -100],
            [4, 5, 6, -100],
        ],
        dtype=torch.long,
    )
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(
        instruction_module,
        "EvaluatorInstructionFineTuning",
        FakeEvaluatorInstructionFineTuning,
    )
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    return TrainerInstructionFineTuning(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )


def test_reset_training_state_clears_history(trainer):
    trainer.train_losses = [1.0]
    trainer.val_losses = [2.0]
    trainer.total_tokens_seen = [5]
    trainer.epochs_eval = [0.5]
    trainer.tokens_seen = 999
    trainer.global_steps = 123

    trainer.reset_training_state()

    assert trainer.train_losses == []
    assert trainer.val_losses == []
    assert trainer.total_tokens_seen == []
    assert trainer.epochs_eval == []
    assert trainer.tokens_seen == 0
    assert trainer.global_steps == -1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"num_epochs": 0}, "num_epochs"),
        ({"freq_evaluation": 0}, "freq_evaluation"),
        ({"progress_update_freq": 0}, "progress_update_freq"),
        ({"freq_checkpoint": 0}, "freq_checkpoint"),
    ],
)
def test_validate_training_configuration_rejects_numeric_invalid_inputs(
    trainer, kwargs, error
):
    base = {
        "num_epochs": 1,
        "freq_evaluation": 1,
        "progress_update_freq": 1,
        "freq_checkpoint": None,
        "learning_rate_scheduler": None,
        "warmup": False,
        "cosine_decay": False,
        "initial_learning_rates": None,
        "peak_learning_rates": None,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        trainer.validate_training_configuration(**base)


def test_validate_training_configuration_requires_scheduler_when_enabled(trainer):
    with pytest.raises(ValueError, match="learning_rate_scheduler must be provided"):
        trainer.validate_training_configuration(
            num_epochs=1,
            freq_evaluation=1,
            progress_update_freq=1,
            freq_checkpoint=None,
            learning_rate_scheduler=None,
            warmup=True,
            cosine_decay=False,
            initial_learning_rates=[0.0],
            peak_learning_rates=[0.01],
        )


def test_validate_training_configuration_requires_scheduler_lrs(trainer):
    scheduler = LearningRateScheduler(num_epochs=1, len_train_dataloader=2)

    with pytest.raises(ValueError, match="required when warmup or cosine_decay"):
        trainer.validate_training_configuration(
            num_epochs=1,
            freq_evaluation=1,
            progress_update_freq=1,
            freq_checkpoint=None,
            learning_rate_scheduler=scheduler,
            warmup=True,
            cosine_decay=False,
            initial_learning_rates=None,
            peak_learning_rates=[0.01],
        )

    with pytest.raises(ValueError, match="initial_learning_rates length must match"):
        trainer.validate_training_configuration(
            num_epochs=1,
            freq_evaluation=1,
            progress_update_freq=1,
            freq_checkpoint=None,
            learning_rate_scheduler=scheduler,
            warmup=True,
            cosine_decay=False,
            initial_learning_rates=[0.0, 0.0],
            peak_learning_rates=[0.01],
        )


def test_validate_training_configuration_accepts_valid_cases(trainer):
    trainer.validate_training_configuration(
        num_epochs=2,
        freq_evaluation=1,
        progress_update_freq=1,
        freq_checkpoint=1,
        learning_rate_scheduler=None,
        warmup=False,
        cosine_decay=False,
        initial_learning_rates=None,
        peak_learning_rates=None,
    )


def test_update_learning_rate_sets_optimizer_groups(trainer):
    scheduler = LearningRateScheduler(num_epochs=1, len_train_dataloader=2)
    scheduler.initialize_learning_rates_warmup(
        warmup_percentage=50,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.01],
    )
    trainer.update_learning_rate(
        warmup=True,
        cosine_decay=False,
        global_steps=1,
        learning_rate_scheduler=scheduler,
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_update_learning_rate_rejects_mismatched_group_count(trainer, monkeypatch):
    class FakeScheduler:
        def get_scheduled_learning_rates(self, warmup, cosine_decay, global_steps):
            del warmup, cosine_decay, global_steps
            return [0.01, 0.02]

    with pytest.raises(ValueError, match="must match optimizer param groups"):
        trainer.update_learning_rate(
            warmup=False,
            cosine_decay=False,
            global_steps=1,
            learning_rate_scheduler=FakeScheduler(),
        )


def test_update_progress_bar_sets_expected_postfix(trainer):
    progress_bar = FakeProgressBar()
    trainer.tokens_seen = 120
    trainer.global_steps = 7
    trainer.update_progress_bar(
        progress_bar=progress_bar,
        epoch_start_time=time.time() - 1.0,
        epoch_start_tokens=20,
        epoch_running_loss=2.5,
        epoch_steps=5,
    )
    assert progress_bar.last_postfix is not None
    assert progress_bar.last_postfix["loss"] == "0.5000"
    assert progress_bar.last_postfix["step"] == "7"


def test_print_methods_emit_expected_output(trainer, capsys):
    trainer.tokens_seen = 100
    trainer.global_steps = 3
    trainer.training_start_time = time.time() - 2.0

    trainer.print_metrics(epoch=0, train_loss=1.2, val_loss=1.5)
    trainer.print_epoch_footer(epoch=0, epoch_avg_loss=1.1, epoch_time_seconds=10.0)

    output = capsys.readouterr().out
    assert "Epoch 01" in output
    assert "Loss (T/V): 1.2000 / 1.5000" in output
    assert "Avg Loss: 1.1000" in output


def test_evaluate_model_records_metrics_and_updates_progress(trainer):
    progress_bar = FakeProgressBar()
    train_loader = make_dataloader()
    val_loader = make_dataloader()
    trainer.global_steps = 3
    trainer.tokens_seen = 12

    trainer.evaluate_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        iter_evaluation=2,
        epoch=0,
        progress_bar=progress_bar,
        step_loss=0.33,
    )

    assert len(trainer.train_losses) == 1
    assert len(trainer.val_losses) == 1
    assert len(trainer.total_tokens_seen) == 1
    assert len(trainer.epochs_eval) == 1
    assert progress_bar.last_postfix["loss"] == "0.3300"


def test_save_checkpoint_wraps_utility(trainer, monkeypatch):
    captured = {}

    def fake_save_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(instruction_module, "save_checkpoint", fake_save_checkpoint)
    trainer.save_checkpoint(model_name="tiny", epoch=2)

    assert captured["model_name"] == "tiny"
    assert captured["epoch"] == 2
    assert captured["model"] is trainer.model
    assert captured["optimizer"] is trainer.optimizer


def test_save_csv_logs_metrics_wraps_utility(trainer, monkeypatch):
    trainer.epochs_eval = [0.0, 1.0]
    trainer.total_tokens_seen = [10, 20]
    trainer.train_losses = [1.0, 0.8]
    trainer.val_losses = [1.1, 0.9]
    captured = {}

    def fake_save_csv_logs(data, name):
        captured["data"] = data
        captured["name"] = name
        return "logs/fake_metrics.csv"

    monkeypatch.setattr(instruction_module, "save_csv_logs", fake_save_csv_logs)
    trainer.save_csv_logs_metrics(model_name="tiny")

    assert captured["name"] == "tiny_instruct_metrics"
    assert captured["data"]["Epochs"] == [0.0, 1.0]
    assert captured["data"]["Training Loss"] == [1.0, 0.8]


@pytest.mark.parametrize(
    ("warmup", "cosine_decay"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_train_runs_all_scheduler_combinations_with_generator_and_side_effects(
    trainer, monkeypatch, warmup, cosine_decay
):
    train_loader = make_dataloader(batch_size=2)  # 2 steps / epoch
    val_loader = make_dataloader(batch_size=2)
    scheduler = LearningRateScheduler(num_epochs=2, len_train_dataloader=len(train_loader))
    scheduler.initial_learning_rates = [0.0]
    scheduler.peak_learning_rates = [0.01]
    scheduler.learning_rate_warmup_percentage = 50 if warmup else None
    scheduler.minimum_learning_rates_percentage = 10 if cosine_decay else None
    if warmup:
        scheduler.initialize_learning_rates_warmup(
            warmup_percentage=scheduler.learning_rate_warmup_percentage,
            initial_learning_rates=scheduler.initial_learning_rates,
            peak_learning_rates=scheduler.peak_learning_rates,
        )
    if cosine_decay:
        scheduler.initialize_learning_rates_cosine_decay(
            minimum_learning_rates_percentage=scheduler.minimum_learning_rates_percentage,
            initial_learning_rates=scheduler.initial_learning_rates,
            peak_learning_rates=scheduler.peak_learning_rates,
        )

    checkpoints = []
    csv_logs = []
    generated_sequences = []

    def fake_save_checkpoint(**kwargs):
        checkpoints.append(kwargs)

    def fake_save_csv_logs(data, name):
        csv_logs.append((name, data))
        return "logs/fake.csv"

    def fake_generate_prompt(generator, prompt, max_token_length):
        del prompt, max_token_length
        generated_sequences.append(generator.generate_numbers(10))

    monkeypatch.setattr(instruction_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(instruction_module, "save_csv_logs", fake_save_csv_logs)
    monkeypatch.setattr(instruction_module, "generate_prompt", fake_generate_prompt)

    train_losses, val_losses, total_tokens_seen = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=2,
        freq_evaluation=1,
        iter_evaluation=2,
        freq_checkpoint=1,
        model_name="tiny_model",
        save_logs=True,
        generator=TinyNumberGenerator(),
        show_progress_bar=False,
        progress_update_freq=1,
        grad_clip=1.0,
        learning_rate_scheduler=scheduler,
        warmup=warmup,
        cosine_decay=cosine_decay,
    )

    assert len(train_losses) == len(val_losses) == len(total_tokens_seen)
    assert len(train_losses) > 0
    assert trainer.global_steps == 3  # 2 epochs * 2 steps - 1
    assert trainer.tokens_seen > 0
    assert len(checkpoints) == 3  # 2 periodic + 1 final
    assert len(csv_logs) == 2  # save_logs=True per epoch
    assert generated_sequences == [list(range(1, 11)), list(range(1, 11))]


def test_train_calls_gradient_clip_when_enabled(trainer, monkeypatch):
    train_loader = make_dataloader(batch_size=2)
    val_loader = make_dataloader(batch_size=2)
    calls = {"count": 0}

    def fake_clip_grad_norm_(parameters, max_norm):
        del parameters, max_norm
        calls["count"] += 1
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)
    scheduler = LearningRateScheduler(num_epochs=1, len_train_dataloader=len(train_loader))
    scheduler.initial_learning_rates = [0.0]
    scheduler.peak_learning_rates = [0.01]
    scheduler.learning_rate_warmup_percentage = None
    scheduler.minimum_learning_rates_percentage = None

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=1,
        freq_evaluation=1,
        iter_evaluation=1,
        show_progress_bar=False,
        progress_update_freq=1,
        grad_clip=0.5,
        learning_rate_scheduler=scheduler,
        warmup=False,
        cosine_decay=False,
    )

    assert calls["count"] == len(train_loader)


def test_train_signature_rejects_removed_lr_arguments(trainer):
    train_loader = make_dataloader(batch_size=2)
    val_loader = make_dataloader(batch_size=2)
    scheduler = LearningRateScheduler(num_epochs=1, len_train_dataloader=len(train_loader))
    scheduler.initial_learning_rates = [0.0]
    scheduler.peak_learning_rates = [0.01]
    scheduler.learning_rate_warmup_percentage = 50
    scheduler.minimum_learning_rates_percentage = None

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            learning_rate_scheduler=scheduler,
            warmup=True,
            learning_rate_warmup_percentage=50,
        )


def test_train_without_scheduler_raises_attribute_error(trainer):
    train_loader = make_dataloader(batch_size=2)
    val_loader = make_dataloader(batch_size=2)

    with pytest.raises(AttributeError, match="initial_learning_rates"):
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            freq_evaluation=1,
            iter_evaluation=1,
            show_progress_bar=False,
            progress_update_freq=1,
            learning_rate_scheduler=None,
            warmup=False,
            cosine_decay=False,
        )
