from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator, Optional, Dict, Any

import torch
from pydantic import BaseModel, Field

from optimizer.tokenizer import Tokenizer, TokenizerConfig
from optimizer.physics_units import PhysicalConstants, UnitSystem
from sensorium.manifold.visualizer import SimulationDashboard
from sensorium.manifold.carriers import CarrierState
from sensorium.observers.base import ObserverProtocol

# Import config classes for type aliases
# These are device-agnostic type aliases that map to device-specific implementations
from optimizer.metal.manifold_physics import (
    ManifoldPhysicsConfig as _ManifoldPhysicsConfig,
    SpectralCarrierConfig as _SpectralCarrierConfig,
)
from dataclasses import dataclass

# Type aliases for unified config interface
@dataclass
class GeometricSimulationConfig:
    """Wrapper for geometric simulation config that includes poisson_iterations for Triton."""
    grid_size: tuple[int, int, int] = (64, 64, 64)
    grid_spacing: float = 1.0
    dt: float = 0.01
    poisson_iterations: int = 50  # Only used for Triton/CUDA
    
    # Material properties (from ManifoldPhysicsConfig)
    particle_radius: float = 0.5
    thermal_conductivity: float = 0.1
    specific_heat: float = 10.0
    dynamic_viscosity: float = 0.01
    emissivity: float = 0.5
    restitution: float = 0.8
    young_modulus: float = 1000.0
    
    # Unit system
    unit_system: UnitSystem = None  # Will be set to default if None
    
    def __post_init__(self):
        if self.unit_system is None:
            self.unit_system = UnitSystem.si()
    
    def to_manifold_physics_config(self, device: str):
        """Convert to ManifoldPhysicsConfig for use with physics engines."""
        if device == "cuda":
            from optimizer.triton.manifold_physics import ManifoldPhysicsConfig as TritonManifoldPhysicsConfig
            # For Triton, poisson_iterations is used
            return TritonManifoldPhysicsConfig(
                grid_size=self.grid_size,
                grid_spacing=self.grid_spacing,
                dt=self.dt,
                poisson_iterations=self.poisson_iterations,
                unit_system=self.unit_system,
                particle_radius=self.particle_radius,
                thermal_conductivity=self.thermal_conductivity,
                specific_heat=self.specific_heat,
                dynamic_viscosity=self.dynamic_viscosity,
                emissivity=self.emissivity,
                restitution=self.restitution,
                young_modulus=self.young_modulus,
            )
        else:
            # For Metal, poisson_iterations is ignored (Metal uses FFT, not iterative)
            config = _ManifoldPhysicsConfig(
                device=device,
                grid_size=self.grid_size,
                grid_spacing=self.grid_spacing,
                dt=self.dt,
                unit_system=self.unit_system,
                particle_radius=self.particle_radius,
                thermal_conductivity=self.thermal_conductivity,
                specific_heat=self.specific_heat,
                dynamic_viscosity=self.dynamic_viscosity,
                emissivity=self.emissivity,
                restitution=self.restitution,
                young_modulus=self.young_modulus,
            )
            return config

@dataclass
class SpectralSimulationConfig:
    """Wrapper for spectral simulation config that includes grid_size and dt."""
    grid_size: tuple[int, int, int] = (64, 64, 64)
    dt: float = 0.01
    poisson_iterations: int = 50  # Ignored (not a spectral parameter, kept for compatibility)
    
    # Spectral carrier config parameters
    max_carriers: int = 64
    coupling_scale: float = 0.25
    carrier_reg: float = 0.15
    conflict_threshold: float = 0.35
    offender_weight_floor: float = 1e-3
    ema_alpha: float = 0.10
    recenter_alpha: float = 0.10
    uncoupled_threshold: float = 0.1
    gate_width_init: float = 0.35
    gate_width_min: float = 0.08
    gate_width_max: float = 1.25
    anchor_slots: int = 8
    stable_amp_threshold: float = 0.25
    crystallize_amp_threshold: float = 0.75
    crystallize_conflict_threshold: float = 0.12
    crystallize_age: int = 120
    crystallized_coupling_boost: float = 1.0
    volatile_decay_mul: float = 0.90
    stable_decay_mul: float = 0.98
    crystallized_decay_mul: float = 1.00
    topdown_phase_scale: float = 0.05
    topdown_energy_scale: float = 0.05
    topdown_random_energy_eps: float = 0.02
    anchor_random_eps: float = 0.05
    repulsion_scale: float = 0.05
    
    def to_spectral_carrier_config(self) -> _SpectralCarrierConfig:
        """Convert to SpectralCarrierConfig for use with physics engines."""
        return _SpectralCarrierConfig(
            max_carriers=self.max_carriers,
            coupling_scale=self.coupling_scale,
            carrier_reg=self.carrier_reg,
            conflict_threshold=self.conflict_threshold,
            offender_weight_floor=self.offender_weight_floor,
            ema_alpha=self.ema_alpha,
            recenter_alpha=self.recenter_alpha,
            uncoupled_threshold=self.uncoupled_threshold,
            gate_width_init=self.gate_width_init,
            gate_width_min=self.gate_width_min,
            gate_width_max=self.gate_width_max,
            anchor_slots=self.anchor_slots,
            stable_amp_threshold=self.stable_amp_threshold,
            crystallize_amp_threshold=self.crystallize_amp_threshold,
            crystallize_conflict_threshold=self.crystallize_conflict_threshold,
            crystallize_age=self.crystallize_age,
            crystallized_coupling_boost=self.crystallized_coupling_boost,
            volatile_decay_mul=self.volatile_decay_mul,
            stable_decay_mul=self.stable_decay_mul,
            crystallized_decay_mul=self.crystallized_decay_mul,
            topdown_phase_scale=self.topdown_phase_scale,
            topdown_energy_scale=self.topdown_energy_scale,
            topdown_random_energy_eps=self.topdown_random_energy_eps,
            anchor_random_eps=self.anchor_random_eps,
            repulsion_scale=self.repulsion_scale,
        )


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.backends.cuda.is_available():
        return "cuda"
    return "cpu"


def build_physics(cfg):
    if cfg.device == "mps":
        from optimizer.metal.manifold_physics import (
            ManifoldPhysics,
            ManifoldPhysicsConfig,
            SpectralCarrierPhysics,
            SpectralCarrierConfig,
        )

        # Convert GeometricSimulationConfig wrapper to ManifoldPhysicsConfig if needed
        if isinstance(cfg.geometric, GeometricSimulationConfig):
            geo_config = cfg.geometric.to_manifold_physics_config("mps")
        else:
            # Already a ManifoldPhysicsConfig, use directly
            geo_config = cfg.geometric
        
        geo = ManifoldPhysics(geo_config)
        # Convert SpectralSimulationConfig wrapper to SpectralCarrierConfig if needed
        if isinstance(cfg.spectral, SpectralSimulationConfig):
            spec_config = cfg.spectral.to_spectral_carrier_config()
        else:
            spec_config = cfg.spectral
        
        spec = SpectralCarrierPhysics(
            spec_config,
            grid_size=cfg.spectral.grid_size,
            dt=cfg.spectral.dt,
        )
        return geo, spec

    if cfg.device == "cuda":
        from optimizer.triton.manifold_physics import (
            ManifoldPhysics,
            SpectralCarrierPhysics,
            SpectralCarrierConfig as TritonSpectralCarrierConfig,
            ManifoldPhysicsConfig as TritonManifoldPhysicsConfig,
        )
        # Convert GeometricSimulationConfig wrapper to ManifoldPhysicsConfig if needed
        if isinstance(cfg.geometric, GeometricSimulationConfig):
            geo_config = cfg.geometric.to_manifold_physics_config("cuda")
        elif isinstance(cfg.geometric, _ManifoldPhysicsConfig):
            # Metal config passed, convert to Triton config
            geo_config = TritonManifoldPhysicsConfig(
                grid_size=cfg.geometric.grid_size,
                grid_spacing=cfg.geometric.grid_spacing,
                dt=cfg.geometric.dt,
                poisson_iterations=50,  # Default for Triton
                unit_system=cfg.geometric.unit_system,
                particle_radius=cfg.geometric.particle_radius,
                thermal_conductivity=cfg.geometric.thermal_conductivity,
                specific_heat=cfg.geometric.specific_heat,
                dynamic_viscosity=cfg.geometric.dynamic_viscosity,
                emissivity=cfg.geometric.emissivity,
                restitution=cfg.geometric.restitution,
                young_modulus=cfg.geometric.young_modulus,
            )
        else:
            # Already a Triton ManifoldPhysicsConfig, use directly
            geo_config = cfg.geometric
        
        # Convert SpectralSimulationConfig wrapper to SpectralCarrierConfig if needed
        if isinstance(cfg.spectral, SpectralSimulationConfig):
            spec_config = TritonSpectralCarrierConfig(
                max_carriers=cfg.spectral.max_carriers,
                coupling_scale=cfg.spectral.coupling_scale,
                carrier_reg=cfg.spectral.carrier_reg,
                temperature=0.01,  # Default for Triton
                conflict_threshold=cfg.spectral.conflict_threshold,
                offender_weight_floor=cfg.spectral.offender_weight_floor,
                ema_alpha=cfg.spectral.ema_alpha,
                recenter_alpha=cfg.spectral.recenter_alpha,
                uncoupled_threshold=cfg.spectral.uncoupled_threshold,
                gate_width_init=cfg.spectral.gate_width_init,
                gate_width_min=cfg.spectral.gate_width_min,
                gate_width_max=cfg.spectral.gate_width_max,
                anchor_slots=cfg.spectral.anchor_slots,
                stable_amp_threshold=cfg.spectral.stable_amp_threshold,
                crystallize_amp_threshold=cfg.spectral.crystallize_amp_threshold,
                crystallize_conflict_threshold=cfg.spectral.crystallize_conflict_threshold,
                crystallize_age=cfg.spectral.crystallize_age,
                crystallized_coupling_boost=cfg.spectral.crystallized_coupling_boost,
                volatile_decay_mul=cfg.spectral.volatile_decay_mul,
                stable_decay_mul=cfg.spectral.stable_decay_mul,
                crystallized_decay_mul=cfg.spectral.crystallized_decay_mul,
                topdown_phase_scale=cfg.spectral.topdown_phase_scale,
                topdown_energy_scale=cfg.spectral.topdown_energy_scale,
                topdown_random_energy_eps=cfg.spectral.topdown_random_energy_eps,
                anchor_random_eps=cfg.spectral.anchor_random_eps,
                repulsion_scale=cfg.spectral.repulsion_scale,
            )
        else:
            spec_config = cfg.spectral
        
        return (
            ManifoldPhysics(geo_config),
            SpectralCarrierPhysics(
                spec_config,
                grid_size=cfg.spectral.grid_size,
                dt=cfg.spectral.dt,
            ),
        )

    raise ValueError(f"Unsupported device: {cfg.device}")


class SimulationConfig(BaseModel):
    tokenizer: TokenizerConfig
    generator: Optional[Callable[[], Iterator[bytes]]] = None

    geometric: GeometricSimulationConfig = Field(default_factory=GeometricSimulationConfig)
    spectral: SpectralSimulationConfig = Field(default_factory=SpectralSimulationConfig)

    physical_constants: PhysicalConstants = Field(
        default_factory=lambda: PhysicalConstants.from_codata_si(UnitSystem.si())
    )

    device: str = Field(default_factory=get_device)
    dashboard: bool = False
    video_path: Path = Path("dashboard.mp4")

    # ------------------------------------------------------------------
    # Initialization policy (integrity / experiment control)
    # ------------------------------------------------------------------
    # NOTE: Historically, we seeded geometric positions deterministically from
    # token_id. That makes "particles cluster by token_id" a tautology unless
    # you explicitly override it. Keep the default for backwards compatibility,
    # but allow experiments to request an independent initialization.
    #
    # - "hash": token_id -> position hash (legacy)
    # - "random": uniform random positions in the grid (independent of token_id)
    position_init: str = "hash"
    position_init_seed: int = 0


def build_initial_state(tokenizer, grid_size, *, position_init: str = "hash", position_init_seed: int = 0):
    pos, vel, ene, heat, exc, mass = [], [], [], [], [], []
    osc_p, osc_w, osc_e = [], [], []
    token_ids_list = []

    gx, gy, gz = grid_size

    for batch in tokenizer.stream():
        p, o = batch["particle"], batch["oscillator"]
        ids = o["token_ids"]
        token_ids_list.append(ids)

        device = ids.device
        init = str(position_init).strip().lower()
        if init in ("hash", "token", "token_hash", "token-hash"):
            positions_batch = torch.stack(
                [
                    (ids * 73856093) % gx,
                    (ids * 19349663) % gy,
                    (ids * 83492791) % gz,
                ],
                dim=1,
            ).to(dtype=torch.float32)
        elif init in ("random", "rand", "uniform"):
            # Independent of token IDs (integrity-friendly).
            # Use a per-run seed so experiments are reproducible.
            seed = int(position_init_seed)
            # Best-effort device seeding without depending on device Generators
            # (MPS generator support is inconsistent across torch versions).
            try:
                torch.manual_seed(seed)
            except Exception:
                pass
            try:
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
            scale = torch.tensor([gx, gy, gz], device=device, dtype=torch.float32)
            positions_batch = torch.rand((len(ids), 3), device=device, dtype=torch.float32) * scale
        else:
            raise ValueError(f"Unknown position_init policy: {position_init!r}")
        pos.append(positions_batch)

        # Initialize velocities: small random velocities to get particles moving
        # This matches the pattern used in ParticleState.random() and SyntheticDataGenerator
        dtype = positions_batch.dtype
        velocities_batch = torch.randn(len(ids), 3, device=device, dtype=dtype) * 0.1
        vel.append(velocities_batch)
        ene.append(p["energies"])
        heat.append(p["heats"])
        exc.append(p["excitations"])
        mass.append(p["masses"])

        osc_p.append(o["phase"])
        osc_w.append(o["omega"])
        osc_e.append(o["energy"])

    if not pos:
        return None

    return dict(
        positions=torch.cat(pos),
        velocities=torch.cat(vel),
        energies=torch.cat(ene),
        heats=torch.cat(heat),
        excitations=torch.cat(exc),
        masses=torch.cat(mass),
        osc_phase=torch.cat(osc_p),
        osc_omega=torch.cat(osc_w),
        osc_energy=torch.cat(osc_e),
        token_ids=torch.cat(token_ids_list) if token_ids_list else torch.empty(0, dtype=torch.int64, device=pos[0].device if pos else torch.device("cpu")),
    )


class Manifold:

    def __init__(
        self,
        cfg: SimulationConfig,
        observers: Optional[Dict[str, ObserverProtocol]] = None,
    ):
        self.cfg = cfg
        self.observers = observers or {}

        tok_cfg = cfg.tokenizer
        if cfg.generator:
            tok_cfg = replace(tok_cfg, generator=cfg.generator)

        self.tokenizer = Tokenizer(tok_cfg)
        self.geometric, self.spectral = build_physics(cfg)

        self.dashboard = (
            SimulationDashboard(
                grid_size=cfg.geometric.grid_size,
                device=cfg.device
            )
            if cfg.dashboard else None
        )

        self.state: Dict[str, torch.Tensor] = {}
        self.carriers: Optional[Dict[str, torch.Tensor]] = None

    def set_generator(self, generator: Callable[[], Iterator[bytes]]):
        self.cfg.generator = generator
        tok_cfg = replace(self.cfg.tokenizer, generator=generator)
        self.tokenizer = Tokenizer(tok_cfg)

    def run(self, settle: bool = True, inference: bool = False):
        if self.dashboard:
            self.dashboard.start_recording(self.cfg.video_path)

        try:
            self.state = build_initial_state(
                self.tokenizer,
                self.cfg.geometric.grid_size,
                position_init=self.cfg.position_init,
                position_init_seed=self.cfg.position_init_seed,
            ) or {}

            if (not self.state or 
                "positions" not in self.state or self.state["positions"].numel() == 0 or
                "velocities" not in self.state or self.state["velocities"].numel() == 0 or
                "energies" not in self.state or self.state["energies"].numel() == 0 or
                "heats" not in self.state or self.state["heats"].numel() == 0 or
                "excitations" not in self.state or self.state["excitations"].numel() == 0 or
                "masses" not in self.state or self.state["masses"].numel() == 0):
                raise ValueError("Failed to build initial state")

            # Check if observers exist and have the required keys
            if not isinstance(self.observers, dict):
                # No observers dict provided, run simulation without observation loop
                self._step_geometric()
                self._step_spectral()
            else:
                geo_observer = self.observers.get("geometric")
                spec_observer = self.observers.get("spectral")
                
                # If no observers provided, run simulation without observation loop
                if not geo_observer and not spec_observer:
                    self._step_geometric()
                    self._step_spectral()
                else:
                    # Run simulation loop with observers
                    geo_done = False
                    spec_done = False
                    
                    while True:
                        out: Dict[str, Any] = {
                            "geometric": self._step_geometric(),
                            "spectral": self._step_spectral()
                        }

                        # Observe if observers exist
                        if geo_observer:
                            geo_obs = geo_observer.observe(out["geometric"])
                            geo_done = geo_obs.get("done_thinking", False) if geo_obs else False
                        
                        if spec_observer:
                            spec_obs = spec_observer.observe(out["spectral"])
                            spec_done = spec_obs.get("done_thinking", False) if spec_obs else False
                        
                        # Break if *either* observer says done.
                        # Many experiments treat each observer as a stopping criterion.
                        # - if both exist: stop when either is done
                        # - if only one exists: stop when it is done
                        should_break = (geo_done if geo_observer else False) or (spec_done if spec_observer else False)
                        if should_break:
                            break

            return self.state
        finally:
            if self.dashboard:
                self.dashboard.close()
                self.dashboard.save(self.cfg.video_path)

    def _step_geometric(self):
        out = self.geometric.step(
            self.state["positions"],
            self.state["velocities"],
            self.state["energies"],
            self.state["heats"],
            self.state["excitations"],
            self.state["masses"],
        )
        if out:
            (
                self.state["positions"],
                self.state["velocities"],
                self.state["energies"],
                self.state["heats"],
                self.state["excitations"],
            ) = out[:5]
        # Always return a dict snapshot for observers (even if the backend returns None).
        return {
            "positions": self.state.get("positions"),
            "velocities": self.state.get("velocities"),
            "energies": self.state.get("energies"),
            "heats": self.state.get("heats"),
            "excitations": self.state.get("excitations"),
            "masses": self.state.get("masses"),
            "token_ids": self.state.get("token_ids"),
        }

    def _step_spectral(self):
        self.carriers = self.spectral.step(
            self.state["osc_phase"],
            self.state["excitations"],
            self.state["energies"],
        )
        return self.carriers
