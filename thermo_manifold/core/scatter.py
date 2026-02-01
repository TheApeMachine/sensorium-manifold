from __future__ import annotations

from typing import Tuple

try:
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency: PyTorch (`torch`).\n\n"
        "This module provides torch-backed scatter helpers.\n"
        "Install it with one of:\n"
        "- `pip install torch`\n"
        "- `uv pip install torch`\n"
    ) from e

try:  # Optional fast path if torch_scatter is installed.
    import torch_scatter  # type: ignore

    _HAS_TORCH_SCATTER = True
except Exception:
    torch_scatter = None  # type: ignore
    _HAS_TORCH_SCATTER = False


def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum `values` grouped by `index` along dim 0."""
    if values.numel() == 0:
        # Preserve trailing dims.
        return torch.zeros((dim_size,) + tuple(values.shape[1:]), device=values.device, dtype=values.dtype)
    out = torch.zeros((dim_size,) + tuple(values.shape[1:]), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def scatter_max(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Max-reduce `values` grouped by `index` along dim 0."""
    if values.numel() == 0:
        return torch.full((dim_size,), float("-inf"), device=values.device, dtype=values.dtype)
    out = torch.full((dim_size,), float("-inf"), device=values.device, dtype=values.dtype)
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        return out
    # Fallback (slower): sort by index then reduce.
    order = torch.argsort(index)
    idx_sorted = index[order]
    val_sorted = values[order]
    out = out.clone()
    # Iterate runs (CPU-friendly fallback).
    start = 0
    while start < idx_sorted.numel():
        j = int(idx_sorted[start].item())
        end = start + 1
        while end < idx_sorted.numel() and int(idx_sorted[end].item()) == j:
            end += 1
        out[j] = torch.max(val_sorted[start:end])
        start = end
    return out


def segment_softmax(logits: torch.Tensor, segment: torch.Tensor, num_segments: int, eps: float) -> torch.Tensor:
    """Softmax within each segment."""
    if logits.numel() == 0:
        return logits
    if _HAS_TORCH_SCATTER:
        max_per, _ = torch_scatter.scatter_max(logits, segment, dim=0, dim_size=num_segments)
        stabilized = logits - max_per[segment]
        ex = torch.exp(stabilized)
        denom = torch_scatter.scatter_add(ex, segment, dim=0, dim_size=num_segments)
        return ex / (denom[segment] + eps)
    max_per = scatter_max(logits, segment, num_segments)
    stabilized = logits - max_per[segment]
    ex = torch.exp(stabilized)
    denom = scatter_sum(ex, segment, num_segments)
    return ex / (denom[segment] + eps)
