from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import Qwen_3_Tokenizer

from .utils import format_instruction_tuning_data


class InstructionDataLoaderFineTuning:
    def __init__(
        self,
        tokenizer: Any | Qwen_3_Tokenizer,
        batch_size: int = 4,
        format_input: Callable[
            [Dict[str, str]], Dict[str, str]
        ] = format_instruction_tuning_data,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = True,
        num_workers: Optional[int] = 0,
        ignore_index: int = -100,
        mask_inputs: Optional[bool] = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.format_input = format_input
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.dataset = None
        self.mask_inputs = mask_inputs
        self.ignore_index = ignore_index

        # Make sure tokenizer has pad_token_id attribute
        assert hasattr(tokenizer, "pad_token_id"), (
            "Tokenizer must have pad_token_id attribute"
        )

        self.pad_token_id = tokenizer.pad_token_id

    class InstructionDataset(Dataset):
        def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer: Any | Qwen_3_Tokenizer,
            format_input: Callable[[Dict[str, str]], Dict[str, str]],
        ):
            self.data = data

            self.encoded_texts = []
            self.input_lengths = []

            for entry in self.data:
                formatted_entry = format_input(entry)

                encoded_input = tokenizer.encode(
                    text=formatted_entry["input"],
                    chat_wrapped=False,
                )
                encoded_output = tokenizer.encode(
                    text=formatted_entry["output"],
                    chat_wrapped=False,
                )
                input_length = len(encoded_input)

                self.encoded_texts.append(encoded_input + encoded_output)
                self.input_lengths.append(input_length)

        def __getitem__(
            self,
            idx: int,
        ):
            return self.encoded_texts[idx], self.input_lengths[idx]

        def __len__(
            self,
        ):
            return len(self.data)

    def custom_collate_function(
        self,
        batch: List[Tuple[List[int], int]],
        pad_token_id: int,
        ignore_index: int = -100,
        device: torch.device = torch.device("cpu"),
        max_length: Optional[int] = 40_960,
        mask_inputs: Optional[bool] = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_max_length = max(len(item) for item, _ in batch) + 1

        inputs_list, targets_list = [], []

        for item, input_length in batch:
            new_item = item.copy()

            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(padded[:-1], dtype=torch.long)
            targets = torch.tensor(padded[1:], dtype=torch.long)

            targets[targets == pad_token_id] = ignore_index

            if mask_inputs and input_length > 1:
                targets[: input_length - 1] = ignore_index

            if max_length is not None:
                inputs = inputs[:max_length]
                targets = targets[:max_length]

            inputs_list.append(inputs)
            targets_list.append(targets)

        inputs = torch.stack(inputs_list).to(device)
        targets = torch.stack(targets_list).to(device)

        return inputs, targets

    # Expected schema per sample:
    # {
    #   "instruction": "<task description>",
    #   "input": "<optional context, use '' if none>",
    #   "output": "<target response>"
    # }
    def create_dataloader(
        self,
        data: List[Dict[str, str]],
        max_length: Optional[int] = 40_960,
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset = self.InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            format_input=self.format_input,
        )

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=lambda x: self.custom_collate_function(
                batch=x,
                pad_token_id=self.pad_token_id,
                ignore_index=self.ignore_index,
                max_length=max_length,
                mask_inputs=self.mask_inputs,
                device=device,
            ),
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return dataloader
