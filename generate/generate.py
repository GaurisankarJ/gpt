from typing import Any, Iterator, Optional, Set

import torch
import torch.nn as nn


class Generator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        context_length: int = 256,
        eos_id: int = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.eos_id = eos_id
        self.eps = 1e-5
        self.device = next(model.parameters()).device

    def text_to_token_ids(
        self,
        text: str,
        allowed_special: Optional[Set[str]] = {"<|endoftext|>"},
    ) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(text, allowed_special=allowed_special)
        )

    def token_ids_to_text(
        self,
        token_ids: torch.Tensor,
    ) -> str:
        return self.tokenizer.decode(token_ids.tolist())

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

    def generate(
        self,
        idx: torch.Tensor,
        max_token_length: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        idx = idx.to(self.device)
        self.model.eval()

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

            if self.eos_id is not None and idx_next.item() == self.eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=-1)

        return idx

    def generate_stream(
        self,
        idx: torch.Tensor,
        max_token_length: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        idx = idx.to(self.device)
        self.model.eval()

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

            yield idx_next

            if self.eos_id is not None and idx_next.item() == self.eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=-1)
