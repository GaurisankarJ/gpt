import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_main_raises_without_mode(run_instruction_script):
    with pytest.raises(ValueError, match="Select at least one mode"):
        run_instruction_script({"train": False, "test": False, "eval": False})


def test_main_train_path_wires_scheduler_and_trainer(run_instruction_script):
    calls = run_instruction_script(
        {
            "train": True,
            "test": False,
            "eval": False,
            "grad_clip": 0.0,
            "warmup": True,
            "cosine_decay": True,
            "initial_learning_rates": [1e-8],
            "peak_learning_rates": [1e-4],
            "learning_rate_warmup_percentage": 25,
            "minimum_learning_rates_percentage": 5,
        }
    )

    assert calls["parse_args_calls"] == 1
    assert calls["trainer_train_calls"] == 1
    assert calls["scheduler_warmup_init_calls"] == 1
    assert calls["scheduler_cosine_init_calls"] == 1
    assert calls["scheduler_warmup_kwargs"]["warmup_percentage"] == 25
    assert calls["scheduler_cosine_kwargs"]["minimum_learning_rates_percentage"] == 5
    assert calls["trainer_train_kwargs"]["grad_clip"] is None
    assert calls["trainer_train_kwargs"]["warmup"] is True
    assert calls["trainer_train_kwargs"]["cosine_decay"] is True


def test_main_train_without_lora_skips_replacement(run_instruction_script):
    calls = run_instruction_script(
        {
            "train": True,
            "test": False,
            "eval": False,
            "lora": False,
        }
    )

    assert calls["replace_lora_calls"] == 0


def test_main_train_without_scheduler_features_skips_schedule_init(run_instruction_script):
    calls = run_instruction_script(
        {
            "train": True,
            "test": False,
            "eval": False,
            "warmup": False,
            "cosine_decay": False,
        }
    )

    assert calls["trainer_train_calls"] == 1
    assert calls["scheduler_warmup_init_calls"] == 0
    assert calls["scheduler_cosine_init_calls"] == 0


def test_main_test_only_path_runs_eval_loss_and_response_generation(
    run_instruction_script,
):
    calls = run_instruction_script(
        {
            "train": False,
            "test": True,
            "eval": False,
            "model_name": "my_model",
            "iter_evaluation": 7,
        }
    )

    assert calls["trainer_train_calls"] == 0
    assert calls["evaluator_init_calls"] == 1
    assert calls["print_eval_losses_calls"] == 1
    assert calls["print_eval_losses_kwargs"]["iter_evaluation"] == 7
    assert calls["create_and_save_response_data_calls"] == 1
    assert calls["create_and_save_response_data_kwargs"]["model_name"] == "my_model"


def test_main_eval_only_calls_evaluate_model_with_test_data_path(
    run_instruction_script,
):
    calls = run_instruction_script(
        {
            "train": False,
            "test": False,
            "eval": True,
            "test_data_path": "tmp_eval.json",
            "evaluation_model": "llama3.2:3b",
        }
    )

    assert calls["evaluate_model_calls"] == 1
    assert calls["evaluate_model_kwargs"]["test_data_path"] == "tmp_eval.json"
    assert calls["evaluate_model_kwargs"]["evaluation_model"] == "llama3.2:3b"


def test_main_all_modes_execute_together(run_instruction_script):
    calls = run_instruction_script({"train": True, "test": True, "eval": True})

    assert calls["trainer_train_calls"] == 1
    assert calls["print_eval_losses_calls"] == 1
    assert calls["create_and_save_response_data_calls"] == 1
    assert calls["evaluate_model_calls"] == 1
    assert calls["evaluate_model_kwargs"]["test_data"][0]["model_response"] == "z"


def test_end_to_end_pipeline_with_one_entry_dataset(monkeypatch, tmp_path):
    dataset_path = tmp_path / "one_entry.json"
    dataset_path.write_text(
        json.dumps([{"instruction": "Say hi", "input": "", "output": "Hi"}]),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        train=True,
        test=True,
        eval=True,
        test_data_path=str(dataset_path),
        model_name="tiny_e2e",
        model_size="0.6B",
        model_type="instruct",
        tokenizer_file_path="tok.json",
        checkpoint_path="ckpt",
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=False,
        dataset_file_path=str(dataset_path),
        shuffle_before_split=False,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        ignore_index=-100,
        mask_inputs=True,
        max_length=32,
        learning_rate=1e-4,
        weight_decay=0.0,
        num_epochs=1,
        freq_evaluation=1,
        iter_evaluation=1,
        save_logs=False,
        show_progress_bar=False,
        progress_update_freq=1,
        grad_clip=1.0,
        freq_checkpoint=None,
        seed=7,
        evaluation_model="llama3.2:3b",
        warmup=True,
        cosine_decay=True,
        initial_learning_rates=[1e-8],
        learning_rate_warmup_percentage=10.0,
        peak_learning_rates=[1e-4],
        minimum_learning_rates_percentage=10.0,
        lora=False,
        lora_alpha=16,
        lora_rank=16,
    )

    calls = {"train": 0, "test": 0, "eval": 0, "split_sizes": None}

    class FakeModel:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.tensor(1.0))

        def parameters(self):
            return [self._param]

        def to(self, _device):
            return self

    class FakeTrainer:
        def __init__(self, model, optimizer, device):
            del model, optimizer, device

        def train(self, **kwargs):
            calls["train"] += 1
            assert kwargs["num_epochs"] == 1

    class FakeScheduler:
        def __init__(self, num_epochs, len_train_dataloader):
            self.initial_learning_rates = None
            self.peak_learning_rates = None
            self.learning_rate_warmup_percentage = None
            self.minimum_learning_rates_percentage = None
            assert num_epochs == 1
            assert len_train_dataloader == 1

        def initialize_learning_rates_warmup(self, **kwargs):
            assert kwargs["warmup_percentage"] == 10.0

        def initialize_learning_rates_cosine_decay(self, **kwargs):
            assert kwargs["minimum_learning_rates_percentage"] == 10.0

    class FakeEvaluator:
        def __init__(self, model, device):
            del model, device

    class FakeTokenizer:
        def __init__(self, **kwargs):
            del kwargs

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

    def fake_parse_args(_defaults):
        return args

    def fake_load_model(model_size, checkpoint_path, device):
        del model_size, checkpoint_path, device
        return FakeModel(), {"num_layers": 1, "context_length": 16}

    def fake_load_and_split_dataset(dataset_file_path, shuffle_before_split, seed):
        del shuffle_before_split, seed
        data = json.loads(Path(dataset_file_path).read_text(encoding="utf-8"))
        train_data = data[:1]
        val_data = data[:1]
        test_data = data[:1]
        calls["split_sizes"] = (len(train_data), len(val_data), len(test_data))
        return train_data, val_data, test_data

    def fake_create_dataloaders(**kwargs):
        del kwargs
        return [("x", "y")], [("x", "y")], [("x", "y")]

    def fake_print_eval_losses(**kwargs):
        del kwargs
        calls["test"] += 1

    def fake_create_and_save_response_data(**kwargs):
        del kwargs
        return [{"input": "Say hi", "output": "Hi", "model_response": "Hi"}]

    def fake_evaluate_model(**kwargs):
        del kwargs
        calls["eval"] += 1
        return [10]

    utils_module = types.ModuleType("scripts.fine_tune_instruction_utils")
    utils_module.parse_args = fake_parse_args
    utils_module.load_model = fake_load_model
    utils_module.load_and_split_dataset = fake_load_and_split_dataset
    utils_module.create_dataloaders = fake_create_dataloaders
    utils_module.print_eval_losses = fake_print_eval_losses
    utils_module.create_and_save_response_data = fake_create_and_save_response_data
    utils_module.evaluate_model = fake_evaluate_model

    evaluation_module = types.ModuleType("evaluation")
    evaluation_module.EvaluatorInstructionFineTuning = FakeEvaluator

    fine_tuning_module = types.ModuleType("fine_tuning")
    fine_tuning_module.TrainerInstructionFineTuning = FakeTrainer
    fine_tuning_module.LearningRateScheduler = FakeScheduler

    generate_module = types.ModuleType("generate")
    generate_module.Generator_Qwen_3 = FakeGenerator

    peft_module = types.ModuleType("parameter_efficient_fine_tuning")
    peft_module.replace_linear_with_lora = lambda model, rank, alpha: None

    tokenizer_module = types.ModuleType("tokenizer")
    tokenizer_module.Qwen_3_Tokenizer = FakeTokenizer

    utils_runtime_module = types.ModuleType("utils")
    utils_runtime_module.get_device = lambda: torch.device("cpu")

    monkeypatch.setitem(
        sys.modules, "scripts.fine_tune_instruction_utils", utils_module
    )
    monkeypatch.setitem(sys.modules, "evaluation", evaluation_module)
    monkeypatch.setitem(sys.modules, "fine_tuning", fine_tuning_module)
    monkeypatch.setitem(sys.modules, "generate", generate_module)
    monkeypatch.setitem(sys.modules, "parameter_efficient_fine_tuning", peft_module)
    monkeypatch.setitem(sys.modules, "tokenizer", tokenizer_module)
    monkeypatch.setitem(sys.modules, "utils", utils_runtime_module)

    runpy.run_module("scripts.fine_tune_instruction", run_name="__main__")

    assert calls["split_sizes"] == (1, 1, 1)
    assert calls["train"] == 1
    assert calls["test"] == 1
    assert calls["eval"] == 1
