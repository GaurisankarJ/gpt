import torch


# RoPE
def compute_rope_params(
    dim_head: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
):
    assert dim_head % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, dim_head, 2, dtype=dtype)[: (dim_head // 2)].float()
            / dim_head
        )
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, dim_head // 2)

    # Expand angles to match the dim_head
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, dim_head)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
):
    # x: (batch_size, num_heads, seq_len, dim_head)
    _, _, seq_len, dim_head = x.shape
    assert dim_head % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : dim_head // 2]  # First half
    x2 = x[..., dim_head // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = (
        cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, seq_len, dim_head)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)
