"""Composable observer system for the Sensorium Manifold.

This package provides a query-like API for observing simulation state.
The system is built on three types of observers:

1. **Primitives**: Extract raw data from simulation state
   - Modes, Particles, Tokens, Oscillators

2. **Transforms**: Filter and reshape observation results
   - Select, Where, TopK, Crystallized, Volatile

3. **Aggregates**: Reduce observations to summary values
   - Count, Total, Mean, Statistics

4. **InferenceObserver**: Orchestrates inference via dark particles
   - Injects query data as invisible particles
   - Observes system reaction
   - Manages dark particle lifecycle

Usage examples:

    # Simple mode observation with fluent API
    result = Modes().observe(state)
    crystallized = result.where(lambda m: m["is_crystallized"])
    top_5 = crystallized.top_k(5, by="amplitude")
    
    # Composable observer pipeline
    observer = TopK(5, by="amplitude")(Crystallized()(Modes()))
    result = observer.observe(state)
    
    # Inference with dark particles
    result = infer(
        b"Hello world",
        Modes(),
        Crystallized(),
        TopK(5, by="amplitude"),
        steps=10,
    ).observe(state, manifold=manifold)
    
    # Statistics on oscillator frequencies
    stats = Statistics("omega")(Oscillators()).observe(state)
"""

# Core types
from .types import (
    ObservationResult,
    ObserverProtocol,
    PARTICLE_FLAG_DARK,
    MODE_VOLATILE,
    MODE_STABLE,
    MODE_CRYSTALLIZED,
    to_numpy,
    get_visible_mask,
)

# Primitive observers
from .primitives import (
    Modes,
    Particles,
    Tokens,
    Oscillators,
)

# Transform observers
from .transforms import (
    Select,
    Where,
    TopK,
    Crystallized,
    Volatile,
)

# Aggregate observers
from .aggregates import (
    Count,
    Total,
    Mean,
    Statistics,
)

# Dark particle system
from .dark import (
    DarkParticleInjector,
    DarkParticleConfig,
    DarkParticleMask,
)

# Inference orchestration
from .inference import (
    InferenceObserver,
    InferenceConfig,
    infer,
    observe_reaction,
)

__all__ = [
    # Core types
    "ObservationResult",
    "ObserverProtocol",
    "PARTICLE_FLAG_DARK",
    "MODE_VOLATILE",
    "MODE_STABLE",
    "MODE_CRYSTALLIZED",
    "to_numpy",
    "get_visible_mask",
    # Primitives
    "Modes",
    "Particles",
    "Tokens",
    "Oscillators",
    # Transforms
    "Select",
    "Where",
    "TopK",
    "Crystallized",
    "Volatile",
    # Aggregates
    "Count",
    "Total",
    "Mean",
    "Statistics",
    # Dark particles
    "DarkParticleInjector",
    "DarkParticleConfig",
    "DarkParticleMask",
    # Inference
    "InferenceObserver",
    "InferenceConfig",
    "infer",
    "observe_reaction",
]
