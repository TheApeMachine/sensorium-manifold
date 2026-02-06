"""Dark particle system for inference.

Dark particles are query particles that perturb the simulation state
but do not persist, couple to carriers, or become visible to observers.
They exist only to cause the system to react, enabling observation of
that reaction without polluting the learned state.
"""

from .injector import DarkParticleInjector, DarkParticleConfig
from .mask import DarkParticleMask

__all__ = [
    "DarkParticleInjector",
    "DarkParticleConfig",
    "DarkParticleMask",
]
