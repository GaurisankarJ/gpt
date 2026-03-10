import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import fine_tune_instruction_utils as utils_mod  # noqa: E402


TEST_DEFAULTS = {
    "train": False,
    "test": False,
    "eval": False,
    "test_data_path": "test_data.json",
    "model_name": "qwen3_0.6b_base",
    "model_size": "0.6B",
    "model_type": "instruct",
    "tokenizer_file_path": "./tokenizer/qwen_3_instruct_tokenizer.json",
    "dataset_file_path": "instruction_tuning_data.json",
    "checkpoint_path": "qwen3_0.6b_base",
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "add_thinking": False,
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


def parse_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return utils_mod.parse_args(TEST_DEFAULTS)


def test_parse_args_defaults(monkeypatch):
    args = parse_cli(monkeypatch, [])
    assert args.eval is False
    assert args.test_data_path == TEST_DEFAULTS["test_data_path"]
    assert args.model_name == TEST_DEFAULTS["model_name"]
    assert args.initial_learning_rates == TEST_DEFAULTS["initial_learning_rates"]
    assert args.peak_learning_rates == TEST_DEFAULTS["peak_learning_rates"]
    assert args.warmup is TEST_DEFAULTS["warmup"]
    assert args.cosine_decay is TEST_DEFAULTS["cosine_decay"]
    assert args.lora is TEST_DEFAULTS["lora"]
    assert args.lora_alpha == TEST_DEFAULTS["lora_alpha"]
    assert args.lora_rank == TEST_DEFAULTS["lora_rank"]
    assert args.save_logs is True


def test_parse_args_overrides_non_boolean_and_list_values(monkeypatch):
    args = parse_cli(
        monkeypatch,
        [
            "--train",
            "--test",
            "--eval",
            "--test_data_path",
            "one.json",
            "--model_name",
            "my_model",
            "--model_size",
            "1.7B",
            "--model_type",
            "base",
            "--tokenizer_file_path",
            "tok.json",
            "--dataset_file_path",
            "data.json",
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
            "--evaluation_model",
            "llama3.1:8b",
            "--initial_learning_rates",
            "1e-8",
            "2e-8",
            "--peak_learning_rates",
            "1e-4",
            "2e-4",
            "--learning_rate_warmup_percentage",
            "15",
            "--minimum_learning_rates_percentage",
            "5",
            "--lora_alpha",
            "32",
            "--lora_rank",
            "8",
        ],
    )
    assert args.train and args.test and args.eval
    assert args.test_data_path == "one.json"
    assert args.model_name == "my_model"
    assert args.batch_size == 2
    assert args.grad_clip == 0.5
    assert args.initial_learning_rates == [1e-8, 2e-8]
    assert args.peak_learning_rates == [1e-4, 2e-4]
    assert args.learning_rate_warmup_percentage == 15
    assert args.minimum_learning_rates_percentage == 5
    assert args.lora_alpha == 32
    assert args.lora_rank == 8


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
        ("warmup", "warmup"),
        ("cosine_decay", "cosine_decay"),
        ("lora", "lora"),
    ],
)
def test_parse_args_boolean_optional_toggles(monkeypatch, flag, attr):
    args_true = parse_cli(monkeypatch, [f"--{flag}"])
    assert getattr(args_true, attr) is True
    args_false = parse_cli(monkeypatch, [f"--no-{flag}"])
    assert getattr(args_false, attr) is False


def test_load_model_without_checkpoint(monkeypatch):
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

    model, config = utils_mod.load_model(model_size="0.6B", checkpoint_path="", device="cpu")
    assert isinstance(model, FakeModel)
    assert config == {"a": 1}


def test_load_model_raises_for_missing_checkpoint(monkeypatch):
    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def to(self, _device):
            return self

    monkeypatch.setattr(utils_mod, "get_qwen3_config", lambda _size: {"a": 1})
    monkeypatch.setattr(utils_mod, "Qwen_3_Model", FakeModel)
    monkeypatch.setattr(utils_mod, "print_model_memory_size", lambda _model: None)
    monkeypatch.setattr(utils_mod.Path, "exists", lambda _path: False)

    with pytest.raises(FileNotFoundError):
        utils_mod.load_model(model_size="0.6B", checkpoint_path="missing", device="cpu")


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
        utils_mod.torch,
        "load",
        lambda _path, map_location=None: {"model_state_dict": {"k": 1}},
    )

    model, config = utils_mod.load_model(
        model_size="0.6B",
        checkpoint_path="existing",
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

    utils_mod.load_model(model_size="0.6B", checkpoint_path="existing", device="cpu")
    assert fake_model.loaded_state_dict == {"k": 2}


def test_load_and_split_dataset_seeded_shuffle_and_sizes(monkeypatch):
    dataset = [{"id": i} for i in range(20)]
    monkeypatch.setattr(utils_mod, "read_json", lambda _: dataset)
    train_a, val_a, test_a = utils_mod.load_and_split_dataset("ignored.json", True, 42)
    train_b, val_b, test_b = utils_mod.load_and_split_dataset("ignored.json", True, 42)

    assert len(train_a) == 17
    assert len(test_a) == 2
    assert len(val_a) == 1
    assert train_a == train_b and val_a == val_b and test_a == test_b


def test_load_and_split_dataset_single_entry(monkeypatch):
    one = [{"id": "single"}]
    monkeypatch.setattr(utils_mod, "read_json", lambda _: one)
    train_data, val_data, test_data = utils_mod.load_and_split_dataset("one.json", False, 0)
    assert train_data == []
    assert test_data == []
    assert val_data == one


def test_load_and_split_dataset_without_shuffle_preserves_order(monkeypatch):
    data = [{"id": i} for i in range(10)]
    monkeypatch.setattr(utils_mod, "read_json", lambda _: data)
    train_data, val_data, test_data = utils_mod.load_and_split_dataset(
        "order.json", False, 123
    )
    reconstructed = train_data + test_data + val_data
    assert reconstructed == data


def test_create_dataloaders_uses_train_and_eval_builders(monkeypatch):
    created = []

    class FakeInstructionDataLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def create_dataloader(self, data, max_length, device):
            return {"data": data, "max_length": max_length, "device": str(device)}

    monkeypatch.setattr(utils_mod, "InstructionDataLoader", FakeInstructionDataLoader)
    train_dl, val_dl, test_dl = utils_mod.create_dataloaders(
        tokenizer="tok",
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        ignore_index=-100,
        mask_inputs=True,
        train_data=[{"id": "t"}],
        val_data=[{"id": "v"}],
        test_data=[{"id": "x"}],
        max_length=64,
        device=torch.device("cpu"),
    )

    assert len(created) == 2
    assert created[0].kwargs["shuffle"] is True
    assert created[1].kwargs["shuffle"] is False
    assert created[1].kwargs["drop_last"] is False
    assert train_dl["data"][0]["id"] == "t"
    assert val_dl["data"][0]["id"] == "v"
    assert test_dl["data"][0]["id"] == "x"


def test_print_eval_losses_calls_each_split_and_prints_table(capsys):
    class FakeEvaluator:
        def __init__(self):
            self.calls = []

        def calculate_loss_dataloader(self, dataloader, num_batches):
            self.calls.append((dataloader, num_batches))
            return {"train": 0.1, "val": 0.2, "test": 0.3}[dataloader]

    evaluator = FakeEvaluator()
    utils_mod.print_eval_losses(
        evaluator=evaluator,
        train_dataloader="train",
        val_dataloader="val",
        test_dataloader="test",
        iter_evaluation=5,
    )
    assert evaluator.calls == [("train", 5), ("val", 5), ("test", 5)]
    output = capsys.readouterr().out
    assert "Dataset" in output and "Training" in output and "Testing" in output


def test_create_and_save_response_data_writes_timestamped_file(monkeypatch):
    class FakeNow:
        @staticmethod
        def strftime(_fmt):
            return "20260308_170000"

    fake_datetime = SimpleNamespace(now=lambda: FakeNow())
    monkeypatch.setattr(utils_mod.datetime, "datetime", fake_datetime)
    monkeypatch.setattr(utils_mod, "tqdm", lambda iterable, total=None: iterable)
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
        def text_to_token_ids(self, text, chat_wrapped):
            assert chat_wrapped is False
            assert text.startswith("in:")
            return torch.tensor([1, 2], dtype=torch.long)

        def generate(self, idx, max_token_length, cache_enabled):
            del idx, max_token_length, cache_enabled
            return torch.tensor([1, 2, 3], dtype=torch.long)

        def token_ids_to_text(self, _token_ids):
            return "in:test-1 completion"

    data = [{"id": "test-1"}, {"id": "test-2"}]
    result = utils_mod.create_and_save_response_data(
        model_name="m1",
        generator=FakeGenerator(),
        test_data=data,
        max_length=16,
        device=torch.device("cpu"),
        num_samples=1,
    )
    assert saved["file_name"] == (
        "instruction_tuning_data_with_response_m1_20260308_170000.json"
    )
    assert result[0]["model_response"] == "completion"
    assert "model_response" not in result[1]


def test_create_and_save_response_data_num_samples_caps_to_dataset_size(monkeypatch):
    monkeypatch.setattr(utils_mod, "tqdm", lambda iterable, total=None: iterable)
    monkeypatch.setattr(
        utils_mod,
        "format_instruction_tuning_data",
        lambda entry: {"input": f"in:{entry['id']}", "output": "out"},
    )
    monkeypatch.setattr(utils_mod, "save_json", lambda data, file_name: None)

    class FakeGenerator:
        def text_to_token_ids(self, text, chat_wrapped):
            del text, chat_wrapped
            return torch.tensor([1, 2], dtype=torch.long)

        def generate(self, idx, max_token_length, cache_enabled):
            del idx, max_token_length, cache_enabled
            return torch.tensor([1, 2, 3], dtype=torch.long)

        def token_ids_to_text(self, _token_ids):
            return "in:x completion"

    test_data = [{"id": "a"}, {"id": "b"}]
    result = utils_mod.create_and_save_response_data(
        model_name="m1",
        generator=FakeGenerator(),
        test_data=test_data,
        max_length=16,
        device=torch.device("cpu"),
        num_samples=99,
    )
    assert "model_response" in result[0]
    assert "model_response" in result[1]


def test_evaluate_model_uses_passed_test_data(monkeypatch):
    calls = {"ollama": 0}

    def fake_check():
        calls["ollama"] += 1

    monkeypatch.setattr(utils_mod, "check_if_ollama_running", fake_check)
    captured = {}

    def fake_generate_model_scores(json_data, model):
        captured["json_data"] = json_data
        captured["model"] = model
        return [9, 8]

    monkeypatch.setattr(utils_mod, "generate_model_scores", fake_generate_model_scores)

    scores = utils_mod.evaluate_model(
        test_data=[{"input": "a"}],
        evaluation_model="llama3.2:3b",
        test_data_path=None,
    )
    assert calls["ollama"] == 1
    assert scores == [9, 8]
    assert captured["json_data"] == [{"input": "a"}]
    assert captured["model"] == "llama3.2:3b"


def test_evaluate_model_reads_test_data_path(monkeypatch):
    monkeypatch.setattr(utils_mod, "check_if_ollama_running", lambda: None)
    monkeypatch.setattr(utils_mod, "read_json", lambda _path: [{"input": "p"}])
    monkeypatch.setattr(utils_mod, "generate_model_scores", lambda json_data, model: [10])

    scores = utils_mod.evaluate_model(
        test_data=[],
        evaluation_model="llama3.2:3b",
        test_data_path="tmp_eval.json",
    )
    assert scores == [10]
