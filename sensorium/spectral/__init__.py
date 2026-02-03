"""Spectral manifolds for frequency-domain thermodynamics."""

from .manifold import SpectralManifold, SpectralOutput
from .unified import UnifiedManifold, UnifiedParticle, UnifiedOutput, Modality

__all__ = [
    "SpectralManifold",
    "SpectralOutput", 
    "UnifiedManifold",
    "UnifiedParticle",
    "UnifiedOutput",
    "Modality",
]
