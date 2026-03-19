import warnings
from typing import Any, Optional, Set

import tiktoken
import torch

DATA_PATH = "data/"


# Get MPS if available
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS device")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        device = torch.device("cuda")
    else:
        print("No GPU device found, falling back to CPU")
        device = torch.device("cpu")

    return device


# Encode/Decode GPT2
def text_to_token_ids(
    text: str,
    tokenizer: Optional[Any] = None,
    allowed_special: Optional[Set[str]] = {"<|endoftext|>"},
) -> torch.Tensor:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    return torch.tensor(tokenizer.encode(text, allowed_special=allowed_special))


def token_ids_to_text(
    token_ids: torch.Tensor,
    tokenizer: Optional[Any] = None,
) -> str:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer.decode(token_ids.tolist())


# Model
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_gpt2_parameters(model: torch.nn.Module) -> str:
    total_params = count_parameters(model)
    out_head = count_parameters(model.out_head)

    return f"{total_params - out_head:,} Parameters"


def calculate_model_memory_size(
    model: torch.nn.Module,
    input_dtype: torch.dtype = torch.float32,
) -> float:
    total_params = 0
    total_grads = 0
    total_buffers = 0

    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


def print_model_memory_size(model: torch.nn.Module) -> None:
    model_memory_size_float32 = calculate_model_memory_size(
        model=model,
        input_dtype=torch.float32,
    )
    model_memory_size_bfloat16 = calculate_model_memory_size(
        model=model,
        input_dtype=torch.bfloat16,
    )

    print(f"float32 (PyTorch default): {model_memory_size_float32:.2f} GB")
    print(f"bfloat16: {model_memory_size_bfloat16:.2f} GB")


def print_generate_stats(
    output_token_ids: torch.Tensor,
    start_time: float,
    end_time: float,
) -> None:
    total_time = end_time - start_time

    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")

    for name, backend in (
        ("CUDA", getattr(torch, "cuda", None)),
        ("XPU", getattr(torch, "xpu", None)),
    ):
        if backend is not None and backend.is_available():
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )
            if hasattr(backend, "synchronize"):
                backend.synchronize()
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024**3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
