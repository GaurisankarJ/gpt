import argparse
import datetime
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_preprocessing import (
    InstructionDataLoader,
    format_instruction_tuning_data,
    read_json,
    save_json,
)
from evaluation import (
    EvaluatorInstructionFineTuning,
    check_if_ollama_running,
    generate_model_scores,
)
from generate import Generator_Qwen_3
from models import Qwen_3_Model, get_qwen3_config
from parameter_efficient_fine_tuning import replace_linear_with_lora
from tokenizer import Qwen_3_Tokenizer
from utils import print_model_memory_size


def parse_args(HYPERPARAMETER_INSTRUCTION_TUNING: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction Fine-Tuning Script")

    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--test", action="store_true", help="Enable testing mode")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["test_data_path"],
        help="Test data path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["model_name"],
        help="Model name",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["model_size"],
        help="Model size",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["model_type"],
        help="Model type",
    )
    parser.add_argument(
        "--tokenizer_file_path",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["tokenizer_file_path"],
        help="Tokenizer file path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["checkpoint_path"],
        help="Optional checkpoint path to load before train/test",
    )
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["apply_chat_template"],
        help="Apply chat template",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["add_generation_prompt"],
        help="Add generation prompt",
    )
    parser.add_argument(
        "--add_thinking",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["add_thinking"],
        help="Add thinking",
    )
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["dataset_file_path"],
        help="Dataset file path",
    )
    parser.add_argument(
        "--shuffle_before_split",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["shuffle_before_split"],
        help="Shuffle dataset before train/val/test split",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["shuffle"],
        help="Shuffle train dataloader",
    )
    parser.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["drop_last"],
        help="Drop last incomplete train batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["num_workers"],
        help="Number of workers",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["ignore_index"],
        help="Ignore index",
    )
    parser.add_argument(
        "--mask_inputs",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["mask_inputs"],
        help="Mask user prompt tokens in labels",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["max_length"],
        help="Max sequence length",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["weight_decay"],
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["num_epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "--freq_evaluation",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["freq_evaluation"],
        help="Frequency of evaluation",
    )
    parser.add_argument(
        "--iter_evaluation",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["iter_evaluation"],
        help="Number of iterations for evaluation",
    )
    parser.add_argument(
        "--save_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training logs",
    )
    parser.add_argument(
        "--show_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["show_progress_bar"],
        help="Show progress bar",
    )
    parser.add_argument(
        "--progress_update_freq",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["progress_update_freq"],
        help="Progress update frequency",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["grad_clip"],
        help="Gradient clipping norm (use <= 0 to disable)",
    )
    parser.add_argument(
        "--freq_checkpoint",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["freq_checkpoint"],
        help="Epoch checkpoint frequency",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["seed"],
        help="Random seed used for dataset shuffle",
    )
    parser.add_argument(
        "--evaluation_model",
        type=str,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["evaluation_model"],
        help="Evaluation model",
    )
    parser.add_argument(
        "--initial_learning_rates",
        type=float,
        nargs="+",
        default=HYPERPARAMETER_INSTRUCTION_TUNING["initial_learning_rates"],
        help="Initial learning rates",
    )
    parser.add_argument(
        "--learning_rate_warmup_percentage",
        type=float,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["learning_rate_warmup_percentage"],
        help="Learning rate warmup percentage",
    )
    parser.add_argument(
        "--peak_learning_rates",
        type=float,
        nargs="+",
        default=HYPERPARAMETER_INSTRUCTION_TUNING["peak_learning_rates"],
        help="Peak learning rates",
    )
    parser.add_argument(
        "--cosine_decay",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["cosine_decay"],
        help="Cosine decay",
    )
    parser.add_argument(
        "--minimum_learning_rates_percentage",
        type=float,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["minimum_learning_rates_percentage"],
        help="Minimum learning rates percentage",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["warmup"],
        help="Warmup",
    )
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["lora"],
        help="LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["lora_alpha"],
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=HYPERPARAMETER_INSTRUCTION_TUNING["lora_rank"],
        help="LoRA rank",
    )

    return parser.parse_args()


def load_model(
    model_size: str,
    checkpoint_path: str,
    device: torch.device,
    mode: bool = False,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 16,
) -> Tuple[Qwen_3_Model, dict]:
    # Get model configuration
    model_config = get_qwen3_config(model_size)

    # Create Qwen 3 model
    model = Qwen_3_Model(**model_config)
    model.to(device)

    if mode and lora:
        replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha)

    # Print model memory size
    print_model_memory_size(model)

    if checkpoint_path:
        checkpoint_path = Path("./checkpoints").joinpath(f"{checkpoint_path}.pth")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

    return model, model_config


def load_and_split_dataset(
    dataset_file_path: str,
    shuffle_before_split: bool,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    dataset = read_json(dataset_file_path)
    data = list(dataset)
    if shuffle_before_split:
        random.Random(seed).shuffle(data)

    train_end = int(len(data) * 0.85)
    test_end = int(len(data) * 0.95)
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]

    print(f"Train data: {len(train_data)}")
    print(f"Val data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")
    print(f"Total data: {len(data)}")

    return train_data, val_data, test_data


def create_dataloaders(
    tokenizer: Qwen_3_Tokenizer,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    ignore_index: int,
    mask_inputs: bool,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    max_length: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Use train dataloader settings for optimization
    train_loader_builder = InstructionDataLoader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        format_input=format_instruction_tuning_data,
        ignore_index=ignore_index,
        mask_inputs=mask_inputs,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    # Use deterministic eval loaders and keep all samples
    eval_loader_builder = InstructionDataLoader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        format_input=format_instruction_tuning_data,
        ignore_index=ignore_index,
        mask_inputs=mask_inputs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    train_dataloader = train_loader_builder.create_dataloader(
        data=train_data,
        max_length=max_length,
        device=device,
    )
    val_dataloader = eval_loader_builder.create_dataloader(
        data=val_data,
        max_length=max_length,
        device=device,
    )
    test_dataloader = eval_loader_builder.create_dataloader(
        data=test_data,
        max_length=max_length,
        device=device,
    )

    return train_dataloader, val_dataloader, test_dataloader


def print_eval_losses(
    evaluator: EvaluatorInstructionFineTuning,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    iter_evaluation: int,
) -> None:
    train_loss = evaluator.calculate_loss_dataloader(
        dataloader=train_dataloader,
        num_batches=iter_evaluation,
    )
    val_loss = evaluator.calculate_loss_dataloader(
        dataloader=val_dataloader,
        num_batches=iter_evaluation,
    )
    test_loss = evaluator.calculate_loss_dataloader(
        dataloader=test_dataloader,
        num_batches=iter_evaluation,
    )

    print(f"\n{'Dataset':<15} | {'Loss':<10}")
    print("-" * 28)
    print(f"{'Training':<15} | {train_loss:>10.4f}")
    print(f"{'Validation':<15} | {val_loss:>10.4f}")
    print(f"{'Testing':<15} | {test_loss:>10.4f}")


def create_and_save_response_data(
    model_name: str,
    generator: Generator_Qwen_3,
    test_data: List[dict],
    max_length: int,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> List[dict]:
    if num_samples is None:
        num_samples = len(test_data)
    num_samples = min(num_samples, len(test_data))

    for i, entry in tqdm(enumerate(test_data[:num_samples]), total=num_samples):
        entry_formatted = format_instruction_tuning_data(entry)
        input_text = generator.text_to_token_ids(
            text=entry_formatted["input"],
            chat_wrapped=False,
        )
        token_ids = generator.generate(
            idx=input_text.unsqueeze(0).to(device),
            max_token_length=max_length,
            cache_enabled=True,
        )
        generated_text = generator.token_ids_to_text(token_ids.squeeze(0))

        response_text = generated_text[len(entry_formatted["input"]) :].strip()

        entry_formatted["model_response"] = response_text
        test_data[i] = entry_formatted

    save_json(
        test_data,
        f"instruction_tuning_data_with_response_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    return test_data


def evaluate_model(
    test_data: List[dict],
    evaluation_model: str,
    test_data_path: Optional[str] = None,
) -> List[int]:
    check_if_ollama_running()

    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = test_data

    scores = generate_model_scores(json_data=test_data, model=evaluation_model)

    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")

    return scores
