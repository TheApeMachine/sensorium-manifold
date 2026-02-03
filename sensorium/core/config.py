from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class PhysicsMedium:
    """Medium/constant registry for the physics engine.

    This is intentionally simple: it exposes the stabilizing constants that were
    previously implicit inside inner loops (ad-hoc normalization / re-scaling).
    """

    # Controls heat retention/cooling (higher -> slower cooling).
    thermal_resistance: float = 1.0
    # Controls drift damping (higher -> slower motion).
    viscosity: float = 1.0
    # Initial/default scale for energetic quantities before EMAs settle.
    baseline_energy_scale: float = 1.0

    # Time constant for slow-moving, global EMA scales (distance/motion/heat/etc).
    scale_tau: float = 5.0
    # Floor to prevent degenerate (near-zero) scales.
    min_scale: float = 1e-6


@dataclass
class PhysicsConfig:
    """Simulation configuration.

    Notes:
    - `dt` is the integration step (a physical simulation timescale, not a tuned ML hyperparameter).
    - `eps` is numerical safety for division/log.
    """

    dt: float = 1e-2
    eps: float = 1e-8
    # Homeostasis time constant (base value; can be modulated by plasticity).
    tau: float = 1.0

    # Exposed physics medium constants.
    medium: PhysicsMedium = field(default_factory=PhysicsMedium)

    # ----------------------------
    # Plastic homeostasis controls
    # ----------------------------
    # If provided with a gate in [0,1], homeostasis becomes slower during mismatch
    # and applies weaker damping (so energetic spikes persist longer).
    homeostasis_tau_gain: float = 0.0
    homeostasis_strength_gain: float = 0.0

    # ----------------------------
    # Dreaming controls (semantic manifolds)
    # ----------------------------
    # Sampling temperature for exploration (higher -> flatter sampling).
    dream_sampling_temperature: float = 1.0
    # Total dream compute/energy budget per idle_think step.
    dream_energy_budget: float = 32.0
    # How strongly global "stress" (homeostasis ratio above 1) increases budget.
    dream_budget_stress_gain: float = 0.0
    # Stop dreaming early when predictive entropy becomes low (confident/settled).
    dream_entropy_stop: float = 0.0

    # ----------------------------
    # Idle pondering controls
    # ----------------------------
    # Minimum idle time before pondering is allowed (seconds).
    idle_think_delay_seconds: float = 60.0

    # ----------------------------
    # Per-carrier homeostasis (semantic manifolds)
    # ----------------------------
    carrier_tau: float = 5.0
