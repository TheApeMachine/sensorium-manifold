"""Metal backend for real-space thermodynamic domain and ω-field wave domain

This implements the physics simulation for the Sensorium Manifold.
Please understand that what is implemented here does not claim to be a full
resolution model, but a utility bounded implementation that represents the
functional border where the system behaves aligned to its intended purpose.
"""

from __future__ import annotations

from .manifold_physics import (
    ThermodynamicsDomain,
    OmegaWaveDomain,
)

__all__ = [
    "ThermodynamicsDomain",
    "OmegaWaveDomain",
]

