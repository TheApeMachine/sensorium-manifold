"""Metal (MPS) fused kernels for Apple Silicon.

This submodule provides a serious, real implementation of a fused DBA (decoupled)
attention *decode* path for MPS, backed by a custom Metal Shading Language kernel
and an Objective-C++ PyTorch extension.

The kernel is intended for autoregressive decode (single-token query) where the
primary cost is repeatedly materializing score tensors and performing unfused
softmax/value matmuls. The Metal kernel performs a numerically-stable, two-pass
softmax (max + exp-sum) fused with the V-weighted reduction, without materializing
the full score matrix.
"""

from __future__ import annotations

from .manifold_physics import ManifoldPhysics, ManifoldPhysicsConfig, manifold_physics_available

__all__ = [
    "ManifoldPhysics",
    "ManifoldPhysicsConfig",
    "manifold_physics_available",
]

