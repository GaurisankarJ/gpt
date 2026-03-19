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
        apply_chat_template: Optional[bool] = None,
        add_generation_prompt: Optional[bool] = None,
        add_thinking: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.eps = 1e-5
        self.device = next(model.parameters()).device

        if model_type == "thinking":
            default_apply = True
            default_gen = True
            default_thinking = True
        elif model_type == "instruct":
            default_apply = True
            default_gen = True
            default_thinking = False
        else:
            default_apply = False
            default_gen = False
            default_thinking = False

        tokenizer = Qwen_3_Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            model_type=model_type,
            apply_chat_template=default_apply
            if apply_chat_template is None
            else apply_chat_template,
            add_generation_prompt=default_gen
            if add_generation_prompt is None
            else add_generation_prompt,
            add_thinking=default_thinking if add_thinking is None else add_thinking,
        )

        self.tokenizer = tokenizer
        self.cache = KVCache(num_layers=num_layers)

    def text_to_token_ids(
        self,
        text: str,
        chat_wrapped: Optional[bool] = None,
    ) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(text=text, chat_wrapped=chat_wrapped),
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
        if not isinstance(top_k, int) or top_k < 1:
            return logits
        k = min(top_k, logits.size(-1))

        top_k_logits, top_k_indices = torch.topk(logits, k=k)

        updated_logits = torch.full_like(logits, -torch.inf, device=logits.device)
        updated_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

        logits = updated_logits

        return logits

    def get_top_p(
        self,
        probs: torch.Tensor,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        if not isinstance(top_p, (float, int)) or not (0 < top_p <= 1):
            return probs

        probs_sorted, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        probs_sorted_cumsum = torch.cumsum(probs_sorted, dim=-1)
        keep_mask = probs_sorted_cumsum - probs_sorted < top_p
        keep_mask[..., 0] = True

        kept_sorted_probs = torch.where(
            keep_mask, probs_sorted, torch.zeros_like(probs_sorted)
        )
        filtered_probs = torch.zeros_like(probs).scatter(
            dim=-1, index=sorted_indices, src=kept_sorted_probs
        )

        denominator = torch.sum(filtered_probs, dim=-1, keepdim=True).clamp_min(1e-12)

        updated_probs = filtered_probs / denominator

        return updated_probs

    def get_probs_temperature(
        self,
        logits: torch.Tensor,
        temperature: float = 1,
    ) -> torch.Tensor:
        logits = logits / (temperature + self.eps)

        probs = torch.softmax(logits, dim=-1)

        return probs

    def get_idx_next(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        probs = None

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
        if top_p is not None:
            probs = self.get_top_p(
                probs=probs if probs is not None else torch.softmax(logits, dim=-1),
                top_p=top_p,
            )

        if probs is not None:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdims=True)

        return idx_next

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_token_length: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
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

                idx_next = self.get_idx_next(
                    logits=logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

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

                idx_next = self.get_idx_next(
                    logits=logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

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
        top_p: Optional[float] = None,
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

                idx_next = self.get_idx_next(
                    logits=logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

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

                logits = self.model(idx_cond)

                logits = logits[:, -1, :]

                idx_next = self.get_idx_next(
                    logits=logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                if self.tokenizer.eos_token_id is not None and torch.all(
                    idx_next == self.tokenizer.eos_token_id
                ):
                    break

                yield idx_next

                idx = torch.cat((idx, idx_next), dim=-1)
