from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate.generate_qwen_3 as gen_mod  # noqa: E402


class FakeTokenizer:
    def __init__(
        self,
        tokenizer_file_path,
        model_type,
        apply_chat_template,
        add_generation_prompt,
        add_thinking,
    ):
        self.tokenizer_file_path = tokenizer_file_path
        self.model_type = model_type
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        self.eos_token_id = 5
        self.last_encode_args = None

    def encode(self, text, chat_wrapped=None):
        self.last_encode_args = {"text": text, "chat_wrapped": chat_wrapped}
        return [1, 2, 3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class FakeKVCache:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class TinyGenModel(nn.Module):
    def __init__(self, vocab_size=8, scripted_tokens=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.vocab_size = vocab_size
        self.scripted_tokens = scripted_tokens or [3, 4, 5]
        self.step = 0
        self.calls = []
        self.reset_calls = 0

    def eval(self):
        return self

    def reset_kv_cache(self):
        self.reset_calls += 1

    def forward(self, idx, cache=None):
        self.calls.append({"shape": tuple(idx.shape), "cache": cache})
        batch_size, seq_len = idx.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=idx.device,
        )

        if cache is not None:
            if seq_len > 1:
                next_token = self.scripted_tokens[0]
            else:
                self.step = min(self.step + 1, len(self.scripted_tokens) - 1)
                next_token = self.scripted_tokens[self.step]
        else:
            next_token = self.scripted_tokens[min(self.step, len(self.scripted_tokens) - 1)]
            self.step = min(self.step + 1, len(self.scripted_tokens) - 1)

        logits[:, -1, next_token] = 10.0
        return logits


@pytest.fixture
def patched_generator_deps(monkeypatch):
    monkeypatch.setattr(gen_mod, "Qwen_3_Tokenizer", FakeTokenizer)
    monkeypatch.setattr(gen_mod, "KVCache", FakeKVCache)


def make_generator(model_type="instruct", **kwargs):
    model = kwargs.pop("model", TinyGenModel())
    return gen_mod.Generator_Qwen_3(
        model=model,
        num_layers=2,
        model_type=model_type,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("thinking", (True, True, True)),
        ("instruct", (True, True, False)),
        ("base", (False, False, False)),
    ],
)
def test_init_uses_expected_defaults_by_model_type(patched_generator_deps, model_type, expected):
    generator = make_generator(model_type=model_type)
    tokenizer = generator.tokenizer
    assert (
        tokenizer.apply_chat_template,
        tokenizer.add_generation_prompt,
        tokenizer.add_thinking,
    ) == expected


def test_init_partial_overrides_keep_model_defaults(patched_generator_deps):
    generator = make_generator(model_type="instruct", add_thinking=True)
    tokenizer = generator.tokenizer
    assert tokenizer.apply_chat_template is True
    assert tokenizer.add_generation_prompt is True
    assert tokenizer.add_thinking is True


def test_text_to_token_ids_and_decode_roundtrip_uses_tokenizer(patched_generator_deps):
    generator = make_generator()
    token_ids = generator.text_to_token_ids("hello", chat_wrapped=False)
    assert token_ids.tolist() == [1, 2, 3]
    assert generator.tokenizer.last_encode_args == {
        "text": "hello",
        "chat_wrapped": False,
    }
    assert generator.token_ids_to_text(torch.tensor([1, 2, 3])) == "1 2 3"


def test_get_top_k_filters_logits(patched_generator_deps):
    generator = make_generator()
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    filtered = generator.get_top_k(logits=logits, top_k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert filtered[0, 1] == 4.0
    assert filtered[0, 3] == 3.0


@pytest.mark.parametrize("top_k", [0, -1, 1.5])
def test_get_top_k_invalid_returns_original_logits(patched_generator_deps, top_k):
    generator = make_generator()
    logits = torch.tensor([[0.2, 0.3]])
    out = generator.get_top_k(logits=logits, top_k=top_k)
    assert torch.equal(out, logits)


def test_get_top_p_filters_and_renormalizes_probs(patched_generator_deps):
    generator = make_generator()
    probs = torch.tensor([[0.50, 0.30, 0.20]])
    filtered = generator.get_top_p(probs=probs, top_p=0.70)
    assert filtered[0, 2] == 0.0
    assert torch.allclose(filtered.sum(dim=-1), torch.ones(1))


@pytest.mark.parametrize("top_p", [0.0, 2.0, "bad"])
def test_get_top_p_invalid_returns_original_probs(patched_generator_deps, top_p):
    generator = make_generator()
    probs = torch.tensor([[0.2, 0.8]])
    out = generator.get_top_p(probs=probs, top_p=top_p)
    assert torch.equal(out, probs)


def test_get_idx_next_greedy_path_without_sampling(patched_generator_deps):
    generator = make_generator()
    logits = torch.tensor([[0.1, 0.5, 0.2]])
    idx_next = generator.get_idx_next(logits=logits, temperature=None, top_k=None, top_p=None)
    assert idx_next.item() == 1


def test_get_idx_next_sampling_path_with_temperature_and_top_p(patched_generator_deps, monkeypatch):
    generator = make_generator()
    captured = {}

    def fake_multinomial(probs, num_samples):
        captured["probs"] = probs
        captured["num_samples"] = num_samples
        return torch.tensor([[2]])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    idx_next = generator.get_idx_next(logits=logits, temperature=0.8, top_k=3, top_p=0.9)
    assert idx_next.item() == 2
    assert captured["num_samples"] == 1
    assert torch.allclose(captured["probs"].sum(dim=-1), torch.ones(1))


def test_generate_with_cache_resets_cache_and_stops_on_eos(patched_generator_deps):
    model = TinyGenModel(scripted_tokens=[3, 4, 5])
    generator = make_generator(model=model)
    idx = torch.tensor([[1, 2]])
    out = generator.generate(idx=idx, max_token_length=10, cache_enabled=True)

    # EOS is stop token and is not appended to output sequence.
    assert out.tolist() == [[1, 2, 3, 4]]
    assert model.reset_calls == 1
    assert generator.cache.reset_calls == 1
    assert model.calls[0]["shape"] == (1, 2)
    assert model.calls[1]["shape"] == (1, 1)


def test_generate_without_cache_path_stops_on_eos(patched_generator_deps):
    model = TinyGenModel(scripted_tokens=[3, 4, 5])
    generator = make_generator(model=model)
    idx = torch.tensor([[1, 2]])
    out = generator.generate(idx=idx, max_token_length=10, cache_enabled=False)
    assert out.tolist() == [[1, 2, 3, 4]]
    assert model.reset_calls == 0


def test_generate_stream_yields_tokens_until_before_eos(patched_generator_deps):
    model = TinyGenModel(scripted_tokens=[3, 4, 5])
    generator = make_generator(model=model)
    idx = torch.tensor([[1, 2]])
    yielded = list(generator.generate_stream(idx=idx, max_token_length=10, cache_enabled=True))
    assert [t.item() for t in yielded] == [3, 4]


def test_generator_end_to_end_with_mocked_inputs(patched_generator_deps):
    model = TinyGenModel(scripted_tokens=[6, 7, 5])
    generator = make_generator(model=model, model_type="base")

    prompt_ids = generator.text_to_token_ids("Explain this.", chat_wrapped=False).unsqueeze(0)
    output_ids = generator.generate(
        idx=prompt_ids,
        max_token_length=6,
        temperature=None,
        top_k=None,
        top_p=None,
        cache_enabled=True,
    )
    decoded = generator.token_ids_to_text(output_ids.squeeze(0))

    # Original mocked prompt ids [1,2,3] + generated [6,7] before EOS.
    assert output_ids.tolist() == [[1, 2, 3, 6, 7]]
    assert decoded == "1 2 3 6 7"
