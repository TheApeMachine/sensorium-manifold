"""Thermo Manifold.

This repo currently has:
- **Active** kernel/experiment code under `sensorium/experiments/` and `sensorium/manifold/`
- **Legacy** research code under `sensorium/old/`

Keep this module intentionally light so importing `sensorium.manifold.*` does not
pull in optional/legacy dependencies.
"""

from __future__ import annotations

__all__ = [
    # Compatibility shims (legacy names live under `sensorium/old/`).
    "PhysicsConfig",
    "SemanticManifold",
    "HierarchicalSemanticManifold",
    "SpectralManifold",
    "BridgeManifold",
]


def __getattr__(name: str):  # pragma: no cover
    # Lazily resolve legacy APIs if someone still imports them.
    if name == "PhysicsConfig":
        from .old.core.config import PhysicsConfig as _PhysicsConfig

        return _PhysicsConfig
    if name == "SemanticManifold":
        from .old.semantic.manifold import SemanticManifold as _SemanticManifold

        return _SemanticManifold
    if name == "HierarchicalSemanticManifold":
        from .old.semantic.hierarchical import (
            HierarchicalSemanticManifold as _HierarchicalSemanticManifold,
        )

        return _HierarchicalSemanticManifold
    if name == "SpectralManifold":
        from .old.spectral.manifold import SpectralManifold as _SpectralManifold

        return _SpectralManifold
    if name == "BridgeManifold":
        from .old.bridge.manifold import BridgeManifold as _BridgeManifold

        return _BridgeManifold
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

