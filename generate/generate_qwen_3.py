from typing import Iterator, Literal, Optional

import torch
import torch.nn as nn

from kv_cache import KVCache
from tokenizer import Qwen_3_Tokenizer


class Generator_Qwen_3:
    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        context_length: int = 40_960,
        tokenizer_file_path: str = "./tokenizer/qwen_3_instruct_tokenizer.json",
        model_type: Literal["thinking", "instruct", "base"] = "instruct",
        model_size: Literal["0.6B", "1.7B", "4B", "8B", "14B", "32B"] = "0.6B",
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.eps = 1e-5
        self.device = next(model.parameters()).device

        if model_type == "thinking":
            repo_id = f"Qwen/Qwen3-{model_size}"
            tokenizer = Qwen_3_Tokenizer(
                tokenizer_file_path=tokenizer_file_path,
                repo_id=repo_id,
                apply_chat_template=True,
                add_generation_prompt=True,
                add_thinking=True,
            )
        elif model_type == "instruct":
            repo_id = f"Qwen/Qwen3-{model_size}"
            tokenizer = Qwen_3_Tokenizer(
                tokenizer_file_path=tokenizer_file_path,
                repo_id=repo_id,
                apply_chat_template=True,
                add_generation_prompt=True,
                add_thinking=False,
            )
        elif model_type == "base":
            repo_id = f"Qwen/Qwen3-{model_size}-Base"
            tokenizer = Qwen_3_Tokenizer(
                tokenizer_file_path=tokenizer_file_path,
                repo_id=repo_id,
                apply_chat_template=False,
                add_generation_prompt=False,
                add_thinking=False,
            )

        self.tokenizer = tokenizer
        self.cache = KVCache(num_layers=num_layers)

    def text_to_token_ids(
        self,
        text: str,
    ) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(text),
            device=self.device,
        )

    def token_ids_to_text(
        self,
        token_ids: torch.Tensor,
    ) -> str:
        return self.tokenizer.decode(
            token_ids.cpu().tolist(),
        )

    def get_top_k(
        self,
        logits: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        if top_k < 1 or type(top_k) is not int:
            return logits
        k = min(top_k, logits.size(-1))

        top_k_logits, top_k_indices = torch.topk(logits, k=k)

        updated_logits = torch.full_like(logits, -torch.inf, device=logits.device)
        updated_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

        logits = updated_logits

        return logits

    def get_probs_temperature(
        self,
        logits: torch.Tensor,
        temperature: float = 1,
    ) -> torch.Tensor:
        logits = logits / (temperature + self.eps)

        probs = torch.softmax(logits, dim=-1)

        return probs

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_token_length: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        cache_enabled: bool = True,
    ) -> torch.Tensor:
        idx = idx.to(self.device)
        self.model.eval()

        if cache_enabled:
            self.model.reset_kv_cache()
            self.cache.reset()

            logits = self.model(idx[:, -self.context_length :], cache=self.cache)

            for _ in range(max_token_length):
                logits = logits[:, -1, :]

                if top_k is not None:
                    logits = self.get_top_k(
                        logits=logits,
                        top_k=top_k,
                    )
                if temperature and temperature > 0:
                    probs = self.get_probs_temperature(
                        logits=logits,
                        temperature=temperature,
                    )
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdims=True)

                if self.tokenizer.eos_token_id is not None and torch.all(
                    idx_next == self.tokenizer.eos_token_id
                ):
                    break

                idx = torch.cat((idx, idx_next), dim=-1)

                logits = self.model(idx_next, cache=self.cache)

            return idx

        else:
            for _ in range(max_token_length):
                idx_cond = idx[:, -self.context_length :]

                logits = self.model(idx_cond)

                logits = logits[:, -1, :]

                if top_k is not None:
                    logits = self.get_top_k(
                        logits=logits,
                        top_k=top_k,
                    )
                if temperature and temperature > 0:
                    probs = self.get_probs_temperature(
                        logits=logits,
                        temperature=temperature,
                    )
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdims=True)

                if self.tokenizer.eos_token_id is not None and torch.all(
                    idx_next == self.tokenizer.eos_token_id
                ):
                    break

                idx = torch.cat((idx, idx_next), dim=-1)

            return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_token_length: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        cache_enabled: bool = True,
    ) -> Iterator[torch.Tensor]:
        idx = idx.to(self.device)
        self.model.eval()

        if cache_enabled:
            self.model.reset_kv_cache()
            self.cache.reset()

            logits = self.model(idx[:, -self.context_length :], cache=self.cache)

            for _ in range(max_token_length):
                logits = logits[:, -1, :]

                if top_k is not None:
                    logits = self.get_top_k(
                        logits=logits,
                        top_k=top_k,
                    )
                if temperature and temperature > 0:
                    probs = self.get_probs_temperature(
                        logits=logits,
                        temperature=temperature,
                    )
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdims=True)

                if self.tokenizer.eos_token_id is not None and torch.all(
                    idx_next == self.tokenizer.eos_token_id
                ):
                    break

                yield idx_next

                idx = torch.cat((idx, idx_next), dim=-1)

                logits = self.model(idx_next, cache=self.cache)

        else:
            for _ in range(max_token_length):
                idx_cond = idx[:, -self.context_length :]

                with torch.no_grad():
                    logits = self.model(idx_cond)

                logits = logits[:, -1, :]

                if top_k is not None:
                    logits = self.get_top_k(
                        logits=logits,
                        top_k=top_k,
                    )
                if temperature and temperature > 0:
                    probs = self.get_probs_temperature(
                        logits=logits,
                        temperature=temperature,
                    )
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdims=True)

                if self.tokenizer.eos_token_id is not None and torch.all(
                    idx_next == self.tokenizer.eos_token_id
                ):
                    break

                yield idx_next

                idx = torch.cat((idx, idx_next), dim=-1)
