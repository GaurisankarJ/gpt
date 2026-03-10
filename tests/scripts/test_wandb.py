from types import SimpleNamespace
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import wandb as wandb_mod  # noqa: E402


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged_payloads = []
        self.logged_artifacts = []
        self.finished = False

    def log(self, payload):
        self.logged_payloads.append(payload)

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


def install_fake_wandb(monkeypatch):
    calls = {"define_metric": [], "watch": []}
    fake_run = FakeRun()
    fake_module = types.ModuleType("wandb")
    fake_module.Artifact = FakeArtifact
    fake_module.Table = FakeTable

    def fake_init(**kwargs):
        calls["init_kwargs"] = kwargs
        return fake_run

    def fake_define_metric(*args, **kwargs):
        calls["define_metric"].append((args, kwargs))

    def fake_watch(model, log, log_freq):
        calls["watch"].append((model, log, log_freq))

    fake_module.init = fake_init
    fake_module.define_metric = fake_define_metric
    fake_module.watch = fake_watch
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return fake_run, calls


def test_build_wandb_config_maps_expected_fields():
    args = SimpleNamespace(
        model_name="m",
        model_size="0.6B",
        model_type="instruct",
        checkpoint_path="ckpt",
        dataset_file_path="data.json",
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.1,
        num_epochs=2,
        freq_evaluation=10,
        iter_evaluation=5,
        max_length=128,
        seed=42,
        warmup=True,
        cosine_decay=True,
        initial_learning_rates=[1e-8],
        peak_learning_rates=[1e-4],
        learning_rate_warmup_percentage=5.0,
        minimum_learning_rates_percentage=30.0,
        lora=True,
        lora_rank=16,
        lora_alpha=16,
        wandb_artifacts=False,
        train=True,
        test=False,
        eval=False,
    )
    config = wandb_mod.build_wandb_config(args)
    assert config["model_name"] == "m"
    assert config["batch_size"] == 4
    assert config["initial_learning_rates"] == [1e-8]
    assert config["train"] is True
    assert config["eval"] is False


def test_count_params_counts_total_and_trainable():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    for param in model[1].parameters():
        param.requires_grad = False

    total, trainable = wandb_mod.count_params(model)
    assert total == 13
    assert trainable == 9


def test_wandb_logger_disabled_is_noop():
    logger = wandb_mod.WandbLogger(
        enabled=False,
        project="p",
        run_name="r",
        entity="e",
        tags=["t"],
        config={},
    )
    logger.log({"x": 1})
    logger.set_summary("k", 1)
    logger.finish()
    assert logger.run is None


def test_wandb_logger_enabled_initializes_and_defines_metrics(monkeypatch):
    fake_run, calls = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True,
        project="proj",
        run_name="run",
        entity="entity",
        tags=["a", "b"],
        config={"x": 1},
    )
    assert logger.run is fake_run
    assert calls["init_kwargs"]["project"] == "proj"
    assert calls["init_kwargs"]["name"] == "run"
    assert calls["init_kwargs"]["entity"] == "entity"
    assert calls["init_kwargs"]["tags"] == ["a", "b"]
    assert len(calls["define_metric"]) == 6


def test_wandb_logger_log_summary_watch_and_finish(monkeypatch):
    fake_run, calls = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True, project=None, run_name=None, entity=None, tags=[], config={}
    )
    model = torch.nn.Linear(2, 2)
    logger.log({"metric": 1.0})
    logger.set_summary("score", 7)
    logger.watch_model(model=model, log_freq=10)
    logger.finish()

    assert fake_run.logged_payloads == [{"metric": 1.0}]
    assert fake_run.summary["score"] == 7
    assert calls["watch"] and calls["watch"][0][2] == 10
    assert fake_run.finished is True


def test_log_dataset_artifact_skips_missing_path(monkeypatch, tmp_path):
    fake_run, _ = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True, project=None, run_name=None, entity=None, tags=[], config={}
    )
    missing = tmp_path / "missing.json"
    logger.log_dataset_artifact(
        dataset_file_path=str(missing),
        model_name="m",
        seed=1,
        shuffle_before_split=False,
    )
    assert fake_run.logged_artifacts == []


def test_log_dataset_artifact_logs_when_file_exists(monkeypatch, tmp_path):
    fake_run, _ = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True, project=None, run_name=None, entity=None, tags=[], config={}
    )
    dataset_path = tmp_path / "data.json"
    dataset_path.write_text("[]", encoding="utf-8")

    logger.log_dataset_artifact(
        dataset_file_path=str(dataset_path),
        model_name="m",
        seed=123,
        shuffle_before_split=True,
    )

    assert len(fake_run.logged_artifacts) == 1
    artifact = fake_run.logged_artifacts[0]
    assert artifact.type == "dataset"
    assert artifact.files == [str(dataset_path)]


def test_log_checkpoint_logs_metrics_and_artifact(monkeypatch, tmp_path):
    fake_run, _ = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True, project=None, run_name=None, entity=None, tags=[], config={}
    )
    ckpt = tmp_path / "m.pth"
    ckpt.write_text("ckpt", encoding="utf-8")
    metrics = {"checkpoint/epoch": 1, "checkpoint/global_step": 12}

    logger.log_checkpoint("m", str(ckpt), metrics)

    assert fake_run.logged_payloads[-1] == metrics
    assert len(fake_run.logged_artifacts) == 1
    assert fake_run.logged_artifacts[0].type == "model"


def test_log_response_table_respects_max_rows(monkeypatch):
    fake_run, _ = install_fake_wandb(monkeypatch)
    logger = wandb_mod.WandbLogger(
        enabled=True, project=None, run_name=None, entity=None, tags=[], config={}
    )
    rows = [
        {"instruction": "i1", "input": "a", "output": "b", "model_response": "c"},
        {"instruction": "i2", "input": "d", "output": "e", "model_response": "f"},
    ]

    logger.log_response_table(rows, max_rows=1)

    payload = fake_run.logged_payloads[-1]
    assert "test/generated_samples" in payload
    table = payload["test/generated_samples"]
    assert isinstance(table, FakeTable)
    assert len(table.rows) == 1


def test_wandb_logger_raises_when_enabled_and_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="wandb is enabled but not installed"):
        wandb_mod.WandbLogger(
            enabled=True,
            project=None,
            run_name=None,
            entity=None,
            tags=[],
            config={},
        )
