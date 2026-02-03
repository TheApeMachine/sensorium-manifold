"""Thermo Manifold: emergent thermodynamic AI primitives."""

from __future__ import annotations

from .core.config import PhysicsConfig

__all__ = [
    "PhysicsConfig",
    # Lazily resolved (torch-backed):
    "SemanticManifold",
    "HierarchicalSemanticManifold",
    "SpectralManifold",
    "BridgeManifold",
]


def __getattr__(name: str):  # pragma: no cover
    # Avoid importing torch unless a torch-backed API is actually accessed.
    if name == "SemanticManifold":
        from .semantic.manifold import SemanticManifold as _SemanticManifold

        return _SemanticManifold
    if name == "HierarchicalSemanticManifold":
        from .semantic.hierarchical import (
            HierarchicalSemanticManifold as _HierarchicalSemanticManifold,
        )

        return _HierarchicalSemanticManifold
    if name == "SpectralManifold":
        from .spectral.manifold import SpectralManifold as _SpectralManifold

        return _SpectralManifold
    if name == "BridgeManifold":
        from .bridge.manifold import BridgeManifold as _BridgeManifold

        return _BridgeManifold
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
