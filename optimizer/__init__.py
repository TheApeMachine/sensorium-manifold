"""Optimizer module: accelerator kernels + backend detection.

This package provides low-level optimizations (Metal on MPS, Triton on CUDA).

Caramba kernel policy:
- Deterministic dispatch to validated backends
- Fail loudly for missing *required* fast paths
"""
from __future__ import annotations

__all__: list[str] = []

# Initialize kernel registry at package import so any missing/invalid kernel backends
# fail loudly before training/inference begins.
from optimizer.kernel_registry import KERNELS as _KERNELS  # noqa: F401
