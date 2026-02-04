from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import Callable

from optimizer.physics_units import UnitSystem


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    
    # Grid and time
    grid_size: tuple[int, int, int] = (32, 32, 32)
    grid_spacing: float = 1.0
    dt: float = 0.01
    poisson_iterations: int = 50         # Need 30-50 for proper pressure solving
    
    # [CHOICE] unit system (simulation units → SI units)
    # [FORMULA] x_SI = x_sim * unit_scale
    # [REASON] enables deriving universal constants from CODATA instead of tuning
    # [NOTES] Default is the identity mapping (1 sim unit == 1 SI unit).
    unit_system: UnitSystem = field(default_factory=UnitSystem.si)
    
    # Material properties
    particle_radius: float = 0.5         # Particle radius
    thermal_conductivity: float = 0.1    # Heat transfer rate
    specific_heat: float = 10.0          # Higher = more thermal inertia/stability
    dynamic_viscosity: float = 0.01      # Low for gas-like behavior
    emissivity: float = 0.5              # Radiation efficiency (0-1)
    restitution: float = 0.8             # Collision elasticity (0-1)
    young_modulus: float = 1000.0        # High to prevent interpenetration
    
    # Simulation
    num_particles: int = 1000
    num_carriers: int = 2
    num_steps: int = 50000

    # Reproducibility (integrity checks)
    seed: int = 0
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    
    # Dashboard
    dashboard_enabled: bool = True
    dashboard_update_interval: int = 50  # Update every N steps

    # Optional: record the live dashboard to a video file (mp4/gif).
    # If set, the visualizer will append frames as it runs and finalize on exit.
    dashboard_video_path: Path | None = None
    dashboard_video_fps: int = 30
    dashboard_video_dpi: int = 150
    
    # Profiling
    profile_enabled: bool = False
    profile_warmup_steps: int = 10
    profile_output_dir: Path = field(default_factory=lambda: Path("artifacts/profiles"))
    
    # Continuous/indefinite mode
    continuous: bool = False
    inject_interval_min: float = 10.0  # seconds
    inject_interval_max: float = 60.0  # seconds
    inject_interval_step: float = 5.0  # quantization step
    inject_particles_min: int = 30
    inject_particles_max: int = 100

    # Scripted injection mode (deterministic, step-based)
    # If set, the simulator will inject according to a JSON script instead of random timing.
    injection_script_path: Path | None = None
    injection_script_loop: bool = False

    # Generator
    generator: Callable = field(default_factory=lambda: None)