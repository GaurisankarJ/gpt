from typing import List, Optional, Tuple

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
    ):
        self.cache = [None] * num_layers

    def get(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.cache[layer_idx]

    def get_all(
        self,
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        return self.cache

    def update(
        self,
        layer_idx: int,
        value: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.cache[layer_idx] = value

    def reset(self) -> None:
        self.cache = [None] * len(self.cache)
