import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import fine_tune_instruction_utils as utils_mod


TEST_DEFAULTS = {
    "model_name": "qwen3_0.6b_base",
    "model_size": "0.6B",
    "model_type": "base",
    "tokenizer_file_path": "./tokenizer/qwen_3_instruct_tokenizer.json",
    "dataset_file_path": "instruction_tuning_data.json",
    "repo_id": "Qwen/Qwen3-0.6B-Base",
    "checkpoint_path": "qwen3_0.6b_base",
    "batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.3,
    "num_epochs": 1,
    "freq_evaluation": 100,
    "iter_evaluation": 50,
    "max_length": 256,
    "ignore_index": -100,
    "mask_inputs": True,
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
    "progress_update_freq": 20,
    "grad_clip": 1.0,
    "show_progress_bar": True,
    "freq_checkpoint": None,
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "add_thinking": False,
    "shuffle_before_split": True,
    "seed": 42,
}


def parse_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return utils_mod.parse_args(TEST_DEFAULTS)


def test_parse_args_defaults(monkeypatch):
    args = parse_cli(monkeypatch, [])

    assert args.train is False
    assert args.test is False
    assert args.model_name == TEST_DEFAULTS["model_name"]
    assert args.model_size == TEST_DEFAULTS["model_size"]
    assert args.model_type == TEST_DEFAULTS["model_type"]
    assert args.tokenizer_file_path == TEST_DEFAULTS["tokenizer_file_path"]
    assert args.dataset_file_path == TEST_DEFAULTS["dataset_file_path"]
    assert args.repo_id == TEST_DEFAULTS["repo_id"]
    assert args.checkpoint_path == TEST_DEFAULTS["checkpoint_path"]
    assert args.batch_size == TEST_DEFAULTS["batch_size"]
    assert args.learning_rate == TEST_DEFAULTS["learning_rate"]
    assert args.weight_decay == TEST_DEFAULTS["weight_decay"]
    assert args.num_epochs == TEST_DEFAULTS["num_epochs"]
    assert args.freq_evaluation == TEST_DEFAULTS["freq_evaluation"]
    assert args.iter_evaluation == TEST_DEFAULTS["iter_evaluation"]
    assert args.max_length == TEST_DEFAULTS["max_length"]
    assert args.ignore_index == TEST_DEFAULTS["ignore_index"]
    assert args.mask_inputs == TEST_DEFAULTS["mask_inputs"]
    assert args.shuffle == TEST_DEFAULTS["shuffle"]
    assert args.drop_last == TEST_DEFAULTS["drop_last"]
    assert args.num_workers == TEST_DEFAULTS["num_workers"]
    assert args.progress_update_freq == TEST_DEFAULTS["progress_update_freq"]
    assert args.grad_clip == TEST_DEFAULTS["grad_clip"]
    assert args.show_progress_bar == TEST_DEFAULTS["show_progress_bar"]
    assert args.freq_checkpoint == TEST_DEFAULTS["freq_checkpoint"]
    assert args.apply_chat_template == TEST_DEFAULTS["apply_chat_template"]
    assert args.add_generation_prompt == TEST_DEFAULTS["add_generation_prompt"]
    assert args.add_thinking == TEST_DEFAULTS["add_thinking"]
    assert args.shuffle_before_split == TEST_DEFAULTS["shuffle_before_split"]
    assert args.seed == TEST_DEFAULTS["seed"]
    assert args.save_logs is True


def test_parse_args_overrides_all_non_boolean_parameters(monkeypatch):
    args = parse_cli(
        monkeypatch,
        [
            "--train",
            "--test",
            "--model_name",
            "my_model",
            "--model_size",
            "1.7B",
            "--model_type",
            "instruct",
            "--tokenizer_file_path",
            "tok.json",
            "--dataset_file_path",
            "data.json",
            "--repo_id",
            "Org/Repo",
            "--checkpoint_path",
            "ckpt_v1",
            "--batch_size",
            "2",
            "--learning_rate",
            "1e-4",
            "--weight_decay",
            "0.01",
            "--num_epochs",
            "3",
            "--freq_evaluation",
            "20",
            "--iter_evaluation",
            "10",
            "--max_length",
            "1024",
            "--ignore_index",
            "-1",
            "--num_workers",
            "2",
            "--progress_update_freq",
            "5",
            "--grad_clip",
            "0.5",
            "--freq_checkpoint",
            "1",
            "--seed",
            "7",
        ],
    )

    assert args.train is True
    assert args.test is True
    assert args.model_name == "my_model"
    assert args.model_size == "1.7B"
    assert args.model_type == "instruct"
    assert args.tokenizer_file_path == "tok.json"
    assert args.dataset_file_path == "data.json"
    assert args.repo_id == "Org/Repo"
    assert args.checkpoint_path == "ckpt_v1"
    assert args.batch_size == 2
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 0.01
    assert args.num_epochs == 3
    assert args.freq_evaluation == 20
    assert args.iter_evaluation == 10
    assert args.max_length == 1024
    assert args.ignore_index == -1
    assert args.num_workers == 2
    assert args.progress_update_freq == 5
    assert args.grad_clip == 0.5
    assert args.freq_checkpoint == 1
    assert args.seed == 7


@pytest.mark.parametrize(
    ("flag", "attr"),
    [
        ("apply_chat_template", "apply_chat_template"),
        ("add_generation_prompt", "add_generation_prompt"),
        ("add_thinking", "add_thinking"),
        ("shuffle_before_split", "shuffle_before_split"),
        ("shuffle", "shuffle"),
        ("drop_last", "drop_last"),
        ("mask_inputs", "mask_inputs"),
        ("save_logs", "save_logs"),
        ("show_progress_bar", "show_progress_bar"),
    ],
)
def test_parse_args_boolean_optional_toggles(monkeypatch, flag, attr):
    args_true = parse_cli(monkeypatch, [f"--{flag}"])
    assert getattr(args_true, attr) is True

    args_false = parse_cli(monkeypatch, [f"--no-{flag}"])
    assert getattr(args_false, attr) is False


def test_load_and_split_dataset_sizes_and_seeded_shuffle(monkeypatch):
    dataset = [{"id": i} for i in range(20)]
    monkeypatch.setattr(utils_mod, "read_json", lambda _: dataset)

    train_a, val_a, test_a = utils_mod.load_and_split_dataset(
        dataset_file_path="ignored.json",
        shuffle_before_split=True,
        seed=42,
    )
    train_b, val_b, test_b = utils_mod.load_and_split_dataset(
        dataset_file_path="ignored.json",
        shuffle_before_split=True,
        seed=42,
    )

    assert len(train_a) == 17
    assert len(test_a) == 2
    assert len(val_a) == 1
    assert train_a == train_b
    assert val_a == val_b
    assert test_a == test_b


def test_load_model_raises_for_missing_checkpoint(monkeypatch):
    class FakeModel:
        def __init__(self, **_kwargs):
            self.loaded_state_dict = None

        def to(self, _device):
            return self

        def load_state_dict(self, state):
            self.loaded_state_dict = state

    monkeypatch.setattr(utils_mod, "get_qwen3_config", lambda _size: {"a": 1})
    monkeypatch.setattr(utils_mod, "Qwen_3_Model", FakeModel)
    monkeypatch.setattr(utils_mod, "print_model_memory_size", lambda _model: None)
    monkeypatch.setattr(utils_mod.Path, "exists", lambda _path: False)

    with pytest.raises(FileNotFoundError):
        utils_mod.load_model(
            model_size="0.6B",
            checkpoint_path="missing_checkpoint",
            device="cpu",
        )


def test_load_model_reads_model_state_dict_key(monkeypatch):
    class FakeModel:
        def __init__(self, **_kwargs):
            self.loaded_state_dict = None

        def to(self, _device):
            return self

        def load_state_dict(self, state):
            self.loaded_state_dict = state

    fake_model = FakeModel()
    monkeypatch.setattr(utils_mod, "get_qwen3_config", lambda _size: {"a": 1})
    monkeypatch.setattr(utils_mod, "Qwen_3_Model", lambda **_kwargs: fake_model)
    monkeypatch.setattr(utils_mod, "print_model_memory_size", lambda _model: None)
    monkeypatch.setattr(utils_mod.Path, "exists", lambda _path: True)
    monkeypatch.setattr(
        utils_mod.torch, "load", lambda _path, map_location=None: {"model_state_dict": {"k": 1}}
    )

    model, config = utils_mod.load_model(
        model_size="0.6B",
        checkpoint_path="existing_checkpoint",
        device="cpu",
    )

    assert model is fake_model
    assert config == {"a": 1}
    assert fake_model.loaded_state_dict == {"k": 1}


def test_load_model_reads_raw_state_dict(monkeypatch):
    class FakeModel:
        def __init__(self, **_kwargs):
            self.loaded_state_dict = None

        def to(self, _device):
            return self

        def load_state_dict(self, state):
            self.loaded_state_dict = state

    fake_model = FakeModel()
    monkeypatch.setattr(utils_mod, "get_qwen3_config", lambda _size: {"a": 1})
    monkeypatch.setattr(utils_mod, "Qwen_3_Model", lambda **_kwargs: fake_model)
    monkeypatch.setattr(utils_mod, "print_model_memory_size", lambda _model: None)
    monkeypatch.setattr(utils_mod.Path, "exists", lambda _path: True)
    monkeypatch.setattr(utils_mod.torch, "load", lambda _path, map_location=None: {"k": 2})

    utils_mod.load_model(
        model_size="0.6B",
        checkpoint_path="existing_checkpoint",
        device="cpu",
    )

    assert fake_model.loaded_state_dict == {"k": 2}


def test_create_and_save_response_data_writes_timestamped_file(monkeypatch):
    class FakeNow:
        @staticmethod
        def strftime(_fmt):
            return "20260308_170000"

    fake_datetime = SimpleNamespace(now=lambda: FakeNow())
    monkeypatch.setattr(utils_mod.datetime, "datetime", fake_datetime)
    monkeypatch.setattr(
        utils_mod, "tqdm", lambda iterable, total=None: iterable
    )
    monkeypatch.setattr(
        utils_mod,
        "format_instruction_tuning_data",
        lambda entry: {"input": f"in:{entry['id']}", "output": "out"},
    )

    saved = {}

    def fake_save_json(data, file_name):
        saved["data"] = data
        saved["file_name"] = file_name

    monkeypatch.setattr(utils_mod, "save_json", fake_save_json)

    class FakeGenerator:
        def text_to_token_ids(self, _text):
            return torch.tensor([1, 2], dtype=torch.long)

        def generate(self, idx, max_token_length, cache_enabled):
            return torch.tensor([1, 2, 3], dtype=torch.long)

        def token_ids_to_text(self, _token_ids):
            return "in:test-1 completion"

    test_data = [{"id": "test-1"}, {"id": "test-2"}]
    utils_mod.create_and_save_response_data(
        model_name="m1",
        generator=FakeGenerator(),
        test_data=test_data,
        max_length=16,
        device=torch.device("cpu"),
    )

    assert saved["file_name"] == "instruction_tuning_data_with_response_m1_20260308_170000.json"
    assert "model_response" in saved["data"][0]
