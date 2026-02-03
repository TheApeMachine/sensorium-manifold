"""Kernel registry and startup validation.

Caramba operates on a strict policy when an accelerator is available:
- Pick the fastest supported kernel path deterministically.
- Validate required kernel backends at startup.
- If a required kernel backend is unavailable, fail loudly with an actionable error.
- Log the chosen performance paths exactly once at initialization.

In CPU-only environments (no CUDA/MPS), the registry initializes in a "no fused
kernels" mode so import-time behavior remains safe and unit tests can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from optimizer.runtime import metal_build_tools_available, metal_supported, triton_supported


@dataclass(frozen=True, slots=True)
class KernelRegistry:
    cuda_available: bool
    mps_available: bool
    triton_available: bool
    metal_supported: bool
    metal_build_tools_available: bool
    metal_ops_loaded: bool


_REGISTRY: KernelRegistry | None = None
_LOGGED: bool = False


def _cuda_device_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        idx = int(torch.cuda.current_device())
        name = str(torch.cuda.get_device_name(idx))
        cap = ".".join(str(x) for x in torch.cuda.get_device_capability(idx))
        return f"{name} (sm_{cap})"
    except Exception as e:
        logger.warning(f"CUDA available but device query failed: {e!r}")
        return "CUDA available (device query failed)"


def initialize_kernels() -> KernelRegistry:
    """Initialize and validate accelerator kernel backends.

    This must run exactly once per process (idempotent).
    """
    global _REGISTRY, _LOGGED
    if _REGISTRY is not None:
        return _REGISTRY

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(torch.backends.mps.is_available())
    # CPU-only mode: no validation, no fused kernels.
    if not cuda_available and not mps_available:
        _REGISTRY = KernelRegistry(
            cuda_available=False,
            mps_available=False,
            triton_available=False,
            metal_supported=bool(metal_supported()),
            metal_build_tools_available=bool(metal_build_tools_available()),
            metal_ops_loaded=False,
        )
        return _REGISTRY

    # NOTE: We no longer hard-require Caramba's custom Triton/Metal kernels at startup.
    # The project now prefers the Hugging Face Kernel Hub (`kernels`) when available,
    # and falls back to PyTorch implementations otherwise.
    metal_ops_loaded = False

    _REGISTRY = KernelRegistry(
        cuda_available=cuda_available,
        mps_available=mps_available,
        triton_available=bool(triton_supported()),
        metal_supported=bool(metal_supported()),
        metal_build_tools_available=bool(metal_build_tools_available()),
        metal_ops_loaded=bool(metal_ops_loaded),
    )


    return _REGISTRY


KERNELS: Final[KernelRegistry] = initialize_kernels()
