"""Core utilities for thermodynamic manifolds.

This package is intentionally lightweight at import-time.

Some modules (e.g. `state`, `scatter`) depend on PyTorch; importing them eagerly
would make `import thermo_manifold` fail in environments where torch is not
installed (even if the caller only needs config/dataclasses).
"""

from __future__ import annotations

from .config import PhysicsConfig, PhysicsMedium
from .diagnostics import SemanticDiagnosticsLogger

__all__ = [
    "PhysicsConfig",
    "PhysicsMedium",
    "SemanticDiagnosticsLogger",
    # Lazily resolved (torch-backed):
    "BatchState",
    "scatter_sum",
    "scatter_max",
    "segment_softmax",
]


def __getattr__(name: str):  # pragma: no cover
    # Lazy imports to avoid importing torch unless needed.
    if name == "BatchState":
        from .state import BatchState as _BatchState

        return _BatchState
    if name in {"scatter_sum", "scatter_max", "segment_softmax"}:
        from .scatter import scatter_max as _scatter_max
        from .scatter import scatter_sum as _scatter_sum
        from .scatter import segment_softmax as _segment_softmax

        return {
            "scatter_sum": _scatter_sum,
            "scatter_max": _scatter_max,
            "segment_softmax": _segment_softmax,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
