from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    
    # Grid and time
    grid_size: tuple[int, int, int] = (32, 32, 32)
    grid_spacing: float = 1.0
    dt: float = 0.01
    poisson_iterations: int = 50         # Need 30-50 for proper pressure solving
    
    # Fundamental physical constants (with correct relative magnitudes)
    # Hierarchy: Gravity << Radiation << Thermal << Elastic
    G: float = 0.001                     # Gravity is weakest (long-range, slow)
    k_B: float = 0.1                     # Moderate thermal pressure
    sigma_SB: float = 1e-5               # Radiation is a slow leak (real σ ≈ 5.67e-8)
    
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
    num_steps: int = 500
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    
    # Dashboard
    dashboard_enabled: bool = True
    dashboard_update_interval: int = 10  # Update every N steps

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
