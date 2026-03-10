import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "fine_tune_instruction.py"


@pytest.fixture
def default_script_args():
    return {
        "train": False,
        "test": False,
        "model_name": "qwen3_0.6b_base",
        "model_size": "0.6B",
        "model_type": "base",
        "checkpoint_path": "qwen3_0.6b_base",
        "tokenizer_file_path": "./tokenizer/qwen_3_instruct_tokenizer.json",
        "repo_id": None,
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "add_thinking": False,
        "dataset_file_path": "instruction_tuning_data.json",
        "batch_size": 4,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 0,
        "ignore_index": -100,
        "mask_inputs": True,
        "max_length": 256,
        "learning_rate": 5e-5,
        "weight_decay": 0.3,
        "num_epochs": 1,
        "freq_evaluation": 100,
        "iter_evaluation": 50,
        "save_logs": True,
        "show_progress_bar": True,
        "progress_update_freq": 20,
        "grad_clip": 1.0,
        "freq_checkpoint": None,
        "shuffle_before_split": True,
        "seed": 42,
    }


@pytest.fixture
def run_instruction_script(monkeypatch, default_script_args):
    def _run(arg_overrides=None):
        args_dict = dict(default_script_args)
        if arg_overrides:
            args_dict.update(arg_overrides)
        args = SimpleNamespace(**args_dict)

        calls = {
            "parse_args_calls": 0,
            "trainer_init_calls": 0,
            "trainer_train_calls": 0,
            "evaluator_init_calls": 0,
            "print_eval_losses_calls": 0,
            "save_json_calls": 0,
            "create_and_save_response_data_calls": 0,
        }

        class FakeModel:
            def __init__(self):
                self._param = torch.nn.Parameter(torch.tensor(1.0))

            def parameters(self):
                return [self._param]

        class FakeTrainer:
            def __init__(self, model, optimizer, device):
                calls["trainer_init_calls"] += 1
                calls["trainer_init"] = {
                    "model": model,
                    "optimizer": optimizer,
                    "device": device,
                }

            def train(self, **kwargs):
                calls["trainer_train_calls"] += 1
                calls["trainer_train_kwargs"] = kwargs

        class FakeEvaluator:
            def __init__(self, model, device):
                calls["evaluator_init_calls"] += 1
                calls["evaluator_init"] = {"model": model, "device": device}

        class FakeGenerator:
            def __init__(
                self,
                model,
                num_layers,
                context_length,
                model_size,
                model_type,
                tokenizer_file_path,
            ):
                calls["generator_init"] = {
                    "model": model,
                    "num_layers": num_layers,
                    "context_length": context_length,
                    "model_size": model_size,
                    "model_type": model_type,
                    "tokenizer_file_path": tokenizer_file_path,
                }

            def text_to_token_ids(self, _text):
                return torch.tensor([1, 2, 3], dtype=torch.long)

            def generate(self, idx, max_token_length, cache_enabled):
                calls["generator_generate"] = {
                    "idx_shape": tuple(idx.shape),
                    "max_token_length": max_token_length,
                    "cache_enabled": cache_enabled,
                }
                return torch.tensor([1, 2, 3, 4], dtype=torch.long)

            def token_ids_to_text(self, _token_ids):
                return "abc completion"

        class FakeTokenizer:
            def __init__(self, **kwargs):
                calls["tokenizer_kwargs"] = kwargs

        def fake_parse_args(_defaults):
            calls["parse_args_calls"] += 1
            return args

        def fake_load_model(model_size, checkpoint_path, device):
            calls["load_model_args"] = {
                "model_size": model_size,
                "checkpoint_path": checkpoint_path,
                "device": device,
            }
            return FakeModel(), {"num_layers": 2, "context_length": 64}

        def fake_load_and_split_dataset(dataset_file_path, shuffle_before_split, seed):
            calls["split_dataset_args"] = {
                "dataset_file_path": dataset_file_path,
                "shuffle_before_split": shuffle_before_split,
                "seed": seed,
            }
            return (
                [{"id": "train-1"}],
                [{"id": "val-1"}],
                [{"id": "test-1"}, {"id": "test-2"}],
            )

        def fake_create_dataloaders(**kwargs):
            calls["create_dataloaders_kwargs"] = kwargs
            return "train_loader", "val_loader", "test_loader"

        def fake_print_eval_losses(**kwargs):
            calls["print_eval_losses_calls"] += 1
            calls["print_eval_losses_kwargs"] = kwargs

        def fake_create_and_save_response_data(
            model_name, generator, test_data, max_length, device
        ):
            calls["create_and_save_response_data_calls"] += 1
            calls["create_and_save_response_data_kwargs"] = {
                "model_name": model_name,
                "generator": generator,
                "test_data": test_data,
                "max_length": max_length,
                "device": device,
            }
            for i, row in enumerate(test_data):
                enriched = dict(row)
                enriched["model_response"] = f"mock_response_{i}"
                test_data[i] = enriched
            fake_save_json(
                data=test_data,
                file_name=f"instruction_tuning_data_with_response_{model_name}.json",
            )

        def fake_format_instruction_tuning_data(entry):
            return {"input": f"in:{entry['id']}", "output": "out"}

        def fake_save_json(data, file_name):
            calls["save_json_calls"] += 1
            calls["save_json_args"] = {"data": data, "file_name": file_name}

        utils_module = types.ModuleType("scripts.fine_tune_instruction_utils")
        utils_module.parse_args = fake_parse_args
        utils_module.load_model = fake_load_model
        utils_module.load_and_split_dataset = fake_load_and_split_dataset
        utils_module.create_dataloaders = fake_create_dataloaders
        utils_module.print_eval_losses = fake_print_eval_losses
        utils_module.create_and_save_response_data = fake_create_and_save_response_data

        data_preprocessing_module = types.ModuleType("data_preprocessing")
        data_preprocessing_module.format_instruction_tuning_data = (
            fake_format_instruction_tuning_data
        )
        data_preprocessing_module.save_json = fake_save_json

        evaluation_module = types.ModuleType("evaluation")
        evaluation_module.EvaluatorInstructionFineTuning = FakeEvaluator

        fine_tuning_module = types.ModuleType("fine_tuning")
        fine_tuning_module.TrainerInstructionFineTuning = FakeTrainer

        generate_module = types.ModuleType("generate")
        generate_module.Generator_Qwen_3 = FakeGenerator

        tokenizer_module = types.ModuleType("tokenizer")
        tokenizer_module.Qwen_3_Tokenizer = FakeTokenizer

        utils_runtime_module = types.ModuleType("utils")
        utils_runtime_module.get_device = lambda: torch.device("cpu")

        monkeypatch.setitem(sys.modules, "scripts.fine_tune_instruction_utils", utils_module)
        monkeypatch.setitem(sys.modules, "data_preprocessing", data_preprocessing_module)
        monkeypatch.setitem(sys.modules, "evaluation", evaluation_module)
        monkeypatch.setitem(sys.modules, "fine_tuning", fine_tuning_module)
        monkeypatch.setitem(sys.modules, "generate", generate_module)
        monkeypatch.setitem(sys.modules, "tokenizer", tokenizer_module)
        monkeypatch.setitem(sys.modules, "utils", utils_runtime_module)

        runpy.run_module("scripts.fine_tune_instruction", run_name="__main__")
        return calls

    return _run
