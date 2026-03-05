from typing import Literal

import torch


# Get the configuration for the Qwen3 model based on the model size
def get_qwen3_config(model_size: Literal["0.6B", "1.7B", "4B", "8B", "14B", "32B"]):
    if model_size == "0.6B":
        return {
            "vocab_size": 151_936,  # Vocabulary size
            "context_length": 40_960,  # Context length that was used to train the model
            "dim_embedding": 1024,  # Embedding dimension
            "num_heads": 16,  # Number of attention heads
            "num_layers": 28,  # Number of layers
            "dim_hidden": 3072,  # Size of the intermediate dimension in FeedForward
            "dim_head": 128,  # Size of the heads in GQA
            "qk_norm": True,  # Whether to normalize queries and keys in GQA
            "num_kv_groups": 8,  # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
            "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        }

    elif model_size == "1.7B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "dim_embedding": 2048,  # 2x larger than above
            "num_heads": 16,
            "num_layers": 28,
            "dim_hidden": 6144,  # 2x larger than above
            "dim_head": 128,
            "qk_norm": True,
            "num_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif model_size == "4B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "dim_embedding": 2560,  # 25% larger than above
            "num_heads": 32,  # 2x larger than above
            "num_layers": 36,  # 29% larger than above
            "dim_hidden": 9728,  # ~3x larger than above
            "dim_head": 128,
            "qk_norm": True,
            "num_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif model_size == "8B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "dim_embedding": 4096,  # 60% larger than above
            "num_heads": 32,
            "num_layers": 36,  # 26% larger than above
            "dim_hidden": 12288,
            "dim_head": 128,
            "qk_norm": True,
            "num_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif model_size == "14B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "dim_embedding": 5120,  # 25% larger than above
            "num_heads": 40,  # 25% larger than above
            "num_layers": 40,  # 11% larger than above
            "dim_hidden": 17408,  # 42% larger than above
            "dim_head": 128,
            "qk_norm": True,
            "num_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif model_size == "32B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "dim_embedding": 5120,
            "num_heads": 64,  # 60% larger than above
            "num_layers": 64,  # 60% larger than above
            "dim_hidden": 25600,  # 47% larger than above
            "dim_head": 128,
            "qk_norm": True,
            "num_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    else:
        raise ValueError(f"{model_size} is not supported.")
