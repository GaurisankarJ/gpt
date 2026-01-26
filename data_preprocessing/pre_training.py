from typing import Any, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset


class BasicDataLoaderPreTraining:
    def __init__(
        self,
        tokenizer: Any,
        batch_size: int = 4,
        context_length: int = 256,
        stride: int = 256,
        allowed_special: Optional[Set[str]] = {"<|endoftext|>"},
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = True,
        num_workers: Optional[int] = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_length = context_length
        self.stride = stride
        self.allowed_special = allowed_special
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    class DatasetPreTraining(Dataset):
        def __init__(
            self,
            text: str,
            tokenizer: Any,
            context_length: int,
            stride: int,
            allowed_special: Optional[Set[str]] = {"<|endoftext|>"},
        ):
            self.input_ids = []
            self.target_ids = []

            token_ids = tokenizer.encode(text, allowed_special=allowed_special)

            for i in range(0, len(token_ids) - context_length, stride):
                input_chunk = token_ids[i : i + context_length]
                target_chunk = token_ids[i + 1 : i + context_length + 1]

                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx: int):
            return self.input_ids[idx], self.target_ids[idx]

    def create_dataloader(
        self,
        text: str,
    ):
        dataset = self.DatasetPreTraining(
            text=text,
            tokenizer=self.tokenizer,
            context_length=self.context_length,
            stride=self.stride,
            allowed_special=self.allowed_special,
        )

        dataloader = DataLoader[Any](
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return dataloader
