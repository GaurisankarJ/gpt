from typing import Any, List, Optional, Set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ClassificationDataLoaderFineTuning:
    def __init__(
        self,
        tokenizer: Any,
        batch_size: int = 4,
        context_length: int = 256,
        pad_token: Optional[Set[str]] = {"<|endoftext|>"},
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = True,
        num_workers: Optional[int] = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_length = context_length
        self.pad_token_id = self.tokenizer.encode(
            pad_token,
            allowed_special={"<|endoftext|>"},
        )[0]
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.dataset = None

    class ClassificationDataset(Dataset):
        def __init__(
            self,
            inputs: List[str],
            targets: List[int],
            tokenizer: Any,
            pad_token_id: int,
            max_length: Optional[int] = None,
        ):
            self.inputs = inputs
            self.targets = targets
            self.encoded_texts = [tokenizer.encode(text) for text in self.inputs]

            if max_length is None:
                self.max_length = self._longest_encoded_length()
            else:
                self.max_length = max_length

                self.encoded_texts = [
                    text[: self.max_length] for text in self.encoded_texts
                ]

            # Add padding
            self.encoded_texts = [
                text + [pad_token_id] * (self.max_length - len(text))
                for text in self.encoded_texts
            ]

        def __getitem__(
            self,
            idx: int,
        ):
            encoded_text = self.encoded_texts[idx]
            label = self.targets[idx]

            return (
                torch.tensor(encoded_text, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
            )

        def __len__(
            self,
        ):
            return len(self.inputs)

        def _longest_encoded_length(
            self,
        ):
            max_length = max(len(encoded_text) for encoded_text in self.encoded_texts)

            return max_length

    def create_dataloader(
        self,
        dataframe: pd.DataFrame,
        max_length: Optional[int] = None,
    ):
        inputs = dataframe["Text"]
        targets = dataframe["Label"]

        self.dataset = self.ClassificationDataset(
            inputs=inputs,
            targets=targets,
            tokenizer=self.tokenizer,
            pad_token_id=self.pad_token_id,
            max_length=max_length,
        )

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return dataloader
