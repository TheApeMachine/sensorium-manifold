"""Manifold APIs.

This package contains:
- **Kernel simulation manifold** utilities (`SimulationConfig`, `run_simulation`, dashboards)
- **Stream manifold** (`Manifold`, `ManifoldConfig`) used by kernel experiments that ingest
  generators of byte chunks and run spectral carriers until a readiness condition.
"""

from __future__ import annotations

__all__ = [
    "SimulationConfig",
    "SimulationDashboard",
    "run_simulation",
    "Manifold",
    "ManifoldConfig",
]


def __getattr__(name: str):  # pragma: no cover
    # Keep imports lazy (torch/matplotlib are optional in some environments).
    if name == "SimulationConfig":
        from .config import SimulationConfig as _SimulationConfig

        return _SimulationConfig
    if name == "run_simulation":
        from .simulator import run_simulation as _run_simulation

        return _run_simulation
    if name == "SimulationDashboard":
        from .visualizer import SimulationDashboard as _SimulationDashboard

        return _SimulationDashboard
    if name == "Manifold":
        from .stream import Manifold as _Manifold

        return _Manifold
    if name == "ManifoldConfig":
        from .stream import ManifoldConfig as _ManifoldConfig

        return _ManifoldConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

