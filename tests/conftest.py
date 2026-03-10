import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def default_script_args():
    return {
        "train": False,
        "test": False,
        "eval": False,
        "test_data_path": "instruction_tuning_data_with_response_model_20260310_000000.json",
        "model_name": "qwen3_0.6b_base",
        "model_size": "0.6B",
        "model_type": "instruct",
        "tokenizer_file_path": "./tokenizer/qwen_3_instruct_tokenizer.json",
        "checkpoint_path": "qwen3_0.6b_base",
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "add_thinking": False,
        "dataset_file_path": "instruction_tuning_data.json",
        "shuffle_before_split": False,
        "batch_size": 8,
        "shuffle": False,
        "drop_last": True,
        "num_workers": 0,
        "ignore_index": -100,
        "mask_inputs": True,
        "max_length": 256,
        "learning_rate": 5e-5,
        "weight_decay": 0.2,
        "num_epochs": 1,
        "freq_evaluation": 25,
        "iter_evaluation": 10,
        "save_logs": True,
        "show_progress_bar": True,
        "progress_update_freq": 20,
        "grad_clip": 1.0,
        "freq_checkpoint": None,
        "seed": 42,
        "evaluation_model": "llama3.2:3b",
        "initial_learning_rates": [5e-8],
        "learning_rate_warmup_percentage": 10.0,
        "peak_learning_rates": [5e-5],
        "cosine_decay": True,
        "minimum_learning_rates_percentage": 10.0,
        "warmup": True,
        "lora": True,
        "lora_alpha": 16,
        "lora_rank": 16,
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
            "trainer_train_calls": 0,
            "evaluator_init_calls": 0,
            "print_eval_losses_calls": 0,
            "create_and_save_response_data_calls": 0,
            "evaluate_model_calls": 0,
            "scheduler_warmup_init_calls": 0,
            "scheduler_cosine_init_calls": 0,
            "replace_lora_calls": 0,
        }

        class FakeModel:
            def __init__(self):
                self._param = torch.nn.Parameter(torch.tensor(1.0))

            def parameters(self):
                return [self._param]

            def to(self, _device):
                return self

        class FakeTrainer:
            def __init__(self, model, optimizer, device):
                calls["trainer_init"] = {
                    "model": model,
                    "optimizer": optimizer,
                    "device": device,
                }

            def train(self, **kwargs):
                calls["trainer_train_calls"] += 1
                calls["trainer_train_kwargs"] = kwargs

        class FakeScheduler:
            def __init__(self, num_epochs, len_train_dataloader):
                calls["scheduler_init"] = {
                    "num_epochs": num_epochs,
                    "len_train_dataloader": len_train_dataloader,
                }
                self.initial_learning_rates = None
                self.peak_learning_rates = None
                self.learning_rate_warmup_percentage = None
                self.minimum_learning_rates_percentage = None

            def initialize_learning_rates_warmup(
                self, warmup_percentage, initial_learning_rates, peak_learning_rates
            ):
                calls["scheduler_warmup_init_calls"] += 1
                calls["scheduler_warmup_kwargs"] = {
                    "warmup_percentage": warmup_percentage,
                    "initial_learning_rates": initial_learning_rates,
                    "peak_learning_rates": peak_learning_rates,
                }

            def initialize_learning_rates_cosine_decay(
                self,
                minimum_learning_rates_percentage,
                initial_learning_rates,
                peak_learning_rates,
            ):
                calls["scheduler_cosine_init_calls"] += 1
                calls["scheduler_cosine_kwargs"] = {
                    "minimum_learning_rates_percentage": minimum_learning_rates_percentage,
                    "initial_learning_rates": initial_learning_rates,
                    "peak_learning_rates": peak_learning_rates,
                }

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
                [{"id": "test-1"}],
            )

        def fake_create_dataloaders(**kwargs):
            calls["create_dataloaders_kwargs"] = kwargs
            return [("x", "y")], [("x", "y")], [("x", "y")]

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
            return [{"input": "x", "output": "y", "model_response": "z"}]

        def fake_evaluate_model(test_data, evaluation_model, test_data_path=None):
            calls["evaluate_model_calls"] += 1
            calls["evaluate_model_kwargs"] = {
                "test_data": test_data,
                "evaluation_model": evaluation_model,
                "test_data_path": test_data_path,
            }
            return [7]

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

        def fake_replace_linear_with_lora(model, rank, alpha):
            del model
            calls["replace_lora_calls"] += 1
            calls["replace_lora_kwargs"] = {"rank": rank, "alpha": alpha}

        peft_module.replace_linear_with_lora = fake_replace_linear_with_lora

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
        return calls

    return _run
