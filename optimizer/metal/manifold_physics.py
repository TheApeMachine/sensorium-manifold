"""Metal manifold physics kernels.

Single implementation using Metal acceleration. No fallbacks.
If Metal is not available or something fails, we raise an exception.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch

if TYPE_CHECKING:
    from torch import Tensor


def manifold_physics_available() -> bool:
    """Check if Metal manifold physics is available."""
    if not torch.backends.mps.is_available():
        return False
    try:
        from .jit import load_caramba_metal_ops
        load_caramba_metal_ops()
        return True
    except Exception:
        return False


@dataclass
class ManifoldPhysicsConfig:
    """Configuration for manifold physics simulation."""
    
    grid_size: tuple[int, int, int] = (64, 64, 64)
    # Simulation parameters
    grid_spacing: float = 1.0            # Length scale (defines unit of length)
    dt: float = 0.01                     # Time step (defines unit of time)
    poisson_iterations: int = 50         # Solver iterations (need 30-50 for 64³ grid)
    
    # Fundamental physical constants (in simulation units)
    # CRITICAL: These must have correct relative magnitudes!
    # 
    # Real-world hierarchy (weakest to strongest):
    #   Gravity << Radiation << Thermal << Elastic
    #
    # G:        Gravity is the WEAKEST force - should not dominate local dynamics
    # sigma_SB: Radiation is a slow heat leak, not instant freeze (real σ = 5.67e-8)
    # k_B:      Thermal fluctuations should be moderate, not explosive
    #
    G: float = 0.001                     # Gravitational constant (weak - long range)
    k_B: float = 0.1                     # Boltzmann constant (moderate thermal pressure)
    sigma_SB: float = 1e-5               # Stefan-Boltzmann constant (slow radiation leak)
    
    # Material properties (define what the "stuff" is made of)
    # 
    # particle_radius:      Defines collision geometry
    # thermal_conductivity: Heat diffusion rate (moderate)
    # specific_heat:        How much energy to change temperature (higher = more stable)
    # dynamic_viscosity:    Drag in medium (low for gas-like, high for liquid-like)
    # emissivity:           Fraction of blackbody radiation (0-1)
    # restitution:          Collision elasticity (0 = sticky, 1 = bouncy)
    # young_modulus:        Contact stiffness (HIGH to prevent interpenetration)
    #
    particle_radius: float = 0.5         # Radius of particles
    thermal_conductivity: float = 0.1    # Heat flow rate (reduced to prevent instability)
    specific_heat: float = 10.0          # Heat capacity (higher = more thermal inertia)
    dynamic_viscosity: float = 0.01      # Low viscosity (gas-like medium)
    emissivity: float = 0.5              # Radiation efficiency (0-1)
    restitution: float = 0.8             # Collision elasticity (0-1)
    young_modulus: float = 1000.0        # High stiffness to prevent overlap


@dataclass
class SpectralCarrierConfig:
    """Configuration for spectral carriers (resonance potential, Langevin flow).

    This layer is an energy-based, non-local coupling mechanism:
    - Oscillators: z_i = A_i e^{iθ_i}, with intrinsic frequency ω_i.
    - Carriers:    C_k = R_k e^{iψ_k}, global modes in frequency space.
    - Tuning:      T_ik = exp(-(ω_i - ω_k)^2 / σ_k^2).
    - Updates follow the gradient of a resonance potential + Langevin noise.
    """

    max_carriers: int = 64
    coupling_scale: float = 0.25
    carrier_reg: float = 0.15      # λ: L2 penalty on |C| (prevents runaway growth)
    temperature: float = 0.01      # Langevin temperature (noise strength)

    # Conflict → split (EMA of phase incoherence)
    conflict_threshold: float = 0.35
    offender_weight_floor: float = 1e-3
    ema_alpha: float = 0.10
    recenter_alpha: float = 0.10
    
    # Uncoupled oscillators spawn their own carrier if total coupling < this
    uncoupled_threshold: float = 0.1

    # Gate width (tolerance / specialization)
    gate_width_init: float = 0.35
    gate_width_min: float = 0.08
    gate_width_max: float = 1.25

    # ------------------------------------------------------------------
    # Memory + top-down bias (carrier-as-chunk store)
    # ------------------------------------------------------------------
    # NOTE: Must match CARRIER_ANCHORS in `manifold_physics.metal`.
    anchor_slots: int = 8

    # Crystallization lifecycle
    stable_amp_threshold: float = 0.25
    crystallize_amp_threshold: float = 0.75
    crystallize_conflict_threshold: float = 0.12
    crystallize_age: int = 120

    # Dynamics modifiers
    crystallized_coupling_boost: float = 1.0
    volatile_decay_mul: float = 0.90
    stable_decay_mul: float = 0.98
    crystallized_decay_mul: float = 1.00

    # Top-down effects
    topdown_phase_scale: float = 0.05
    topdown_energy_scale: float = 0.05
    topdown_random_energy_eps: float = 0.02

    # Anchor refresh ε (online); higher values are used in exploration modes
    anchor_random_eps: float = 0.05

    # Idle-time disambiguation: ω-space repulsion between nearby carriers
    repulsion_scale: float = 0.05

    # Genesis: first oscillator creates first carrier (seed_carriers is deprecated)
    seed_carriers: int = 1  # Unused - genesis always starts with exactly 1 carrier


class ManifoldPhysics:
    """Metal-accelerated manifold physics simulation.
    
    No fallbacks. Metal only. Exceptions on failure.
    """
    
    def __init__(
        self,
        config: ManifoldPhysicsConfig,
        device: str = "mps",
    ):
        if device != "mps":
            raise RuntimeError(f"ManifoldPhysics requires device='mps', got '{device}'")
        
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")
        
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32
        
        gx, gy, gz = config.grid_size
        self.grid_dims = (gx, gy, gz)
        
        # Allocate fields
        self.gravity_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.gravity_potential = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.temperature_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        
        # Load Metal ops immediately (fail fast)
        from .jit import load_caramba_metal_ops
        self._ops = load_caramba_metal_ops()
    
    @property
    def ops(self):
        return self._ops
    
    def scatter_particles(
        self,
        positions: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
    ) -> None:
        """Scatter particle mass and heat to fields."""
        self.ops.manifold_clear_field(self.gravity_field)
        self.ops.manifold_clear_field(self.temperature_field)
        self.ops.manifold_scatter_particles(
            positions.contiguous(),
            masses.contiguous(),
            heats.contiguous(),
            self.gravity_field,
            self.temperature_field,
            float(self.config.grid_spacing),
        )
    
    def solve_gravity(self) -> None:
        """Solve Poisson equation for gravitational potential.
        
        Poisson equation: ∇²φ = 4πGρ
        We use G from the config as the gravitational constant.
        """
        cfg = self.config
        # Poisson equation coefficient: 4πG
        gravity_4pi = 4.0 * 3.14159265359 * cfg.G
        
        phi_tmp = torch.zeros_like(self.gravity_potential)
        
        for i in range(cfg.poisson_iterations):
            if i % 2 == 0:
                self.ops.manifold_poisson_step(
                    self.gravity_potential, self.gravity_field, phi_tmp,
                    gravity_4pi, float(cfg.grid_spacing)
                )
            else:
                self.ops.manifold_poisson_step(
                    phi_tmp, self.gravity_field, self.gravity_potential,
                    gravity_4pi, float(cfg.grid_spacing)
                )
        
        if cfg.poisson_iterations % 2 == 1:
            self.gravity_potential.copy_(phi_tmp)
    
    def diffuse_heat(self) -> None:
        """Evolve temperature field via diffusion.
        
        Heat equation: ∂T/∂t = α∇²T where α = k/(ρ*c_v)
        We use thermal_conductivity from config as the diffusion coefficient.
        """
        cfg = self.config
        temp_out = torch.zeros_like(self.temperature_field)
        # Thermal diffusivity α = k (using k directly as we're in simulation units)
        self.ops.manifold_diffuse_heat(
            self.temperature_field, temp_out,
            float(cfg.thermal_conductivity), float(cfg.dt), float(cfg.grid_spacing)
        )
        self.temperature_field.copy_(temp_out)
    
    def gather_update_particles(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        """Gather from fields and update all particle state.
        
        Uses fundamental physics:
        - Gravity: F = -G * m * ∇φ
        - Pressure: F = -k_B * m * ∇T (ideal gas law)
        - Heat transfer: Newton's law + Stefan-Boltzmann radiation
        - Drag: Stokes' law F = -6πηrv
        """
        cfg = self.config
        
        positions = positions.contiguous()
        velocities = velocities.contiguous()
        energies = energies.contiguous()
        heats = heats.contiguous()
        excitations = excitations.contiguous()
        masses = masses.contiguous()
        
        self.ops.manifold_gather_update_particles(
            self.gravity_potential,
            self.temperature_field,
            positions,
            velocities,
            energies,
            heats,
            excitations,
            masses,
            float(cfg.dt),
            float(cfg.grid_spacing),
            # Fundamental constants
            float(cfg.G),
            float(cfg.k_B),
            float(cfg.sigma_SB),
            # Material properties
            float(cfg.particle_radius),
            float(cfg.thermal_conductivity),
            float(cfg.specific_heat),
            float(cfg.dynamic_viscosity),
            float(cfg.emissivity),
            float(cfg.young_modulus),
        )
        return positions, velocities, energies, heats, excitations

    
    def compute_interactions(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor"]:
        """Compute particle-particle interactions (collision + heat transfer).
        
        Implements proper physics:
        - Momentum-conserving impulse-based collisions
        - Energy conservation: KE_lost = 0.5 * m_eff * v_n² * (1 - e²) → heat
        - Hertzian contact force: F = E * δ (prevents interpenetration)
        - Fourier's law heat conduction on contact
        
        This is O(N²) so best for moderate particle counts (<1000).
        """
        cfg = self.config
        
        positions = positions.contiguous()
        velocities = velocities.contiguous()
        excitations = excitations.contiguous()
        masses = masses.contiguous()
        heats = heats.contiguous()
        
        self.ops.particle_interactions(
            positions,
            velocities,
            excitations,
            masses,
            heats,
            float(cfg.dt),
            float(cfg.particle_radius),
            float(cfg.young_modulus),
            float(cfg.thermal_conductivity),
            float(cfg.restitution),
        )
        return velocities, excitations, heats
    
    def compute_interactions_spatial_hash(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
        cell_size: Optional[float] = None,
    ) -> tuple["Tensor", "Tensor", "Tensor"]:
        """Compute particle-particle interactions using spatial hashing.
        
        This is O(N) on average, much faster than O(N²) for large particle counts.
        Use this method when num_particles > 1000.
        
        The simulation domain is divided into cells of size `cell_size`.
        Particles only check for collisions with particles in the same or
        neighboring cells (27 cells total in 3D).
        
        Args:
            cell_size: Size of spatial hash cells. Should be >= 2 * particle_radius.
                       If None, defaults to 4 * particle_radius for optimal performance.
        """
        cfg = self.config
        n = positions.size(0)
        
        if n == 0:
            return velocities, excitations, heats
        
        # Use 4x particle radius if not specified (empirically optimal)
        if cell_size is None:
            cell_size = 4.0 * cfg.particle_radius
        
        # Compute grid dimensions for spatial hash
        gx, gy, gz = self.grid_dims
        domain_size = (
            gx * cfg.grid_spacing,
            gy * cfg.grid_spacing,
            gz * cfg.grid_spacing,
        )
        
        hash_grid_x = max(1, int(math.ceil(domain_size[0] / cell_size)))
        hash_grid_y = max(1, int(math.ceil(domain_size[1] / cell_size)))
        hash_grid_z = max(1, int(math.ceil(domain_size[2] / cell_size)))
        num_cells = hash_grid_x * hash_grid_y * hash_grid_z
        
        # Allocate working buffers (reuse if possible in hot path)
        particle_cell_idx = torch.empty(n, dtype=torch.int32, device=self.device)
        cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=self.device)
        cell_starts = torch.empty(num_cells + 1, dtype=torch.int32, device=self.device)
        sorted_particle_idx = torch.empty(n, dtype=torch.int32, device=self.device)
        
        positions = positions.contiguous()
        velocities = velocities.contiguous()
        excitations = excitations.contiguous()
        masses = masses.contiguous()
        heats = heats.contiguous()
        
        # Phase 1: Assign particles to cells
        self.ops.spatial_hash_assign(
            positions,
            particle_cell_idx,
            cell_counts,
            hash_grid_x,
            hash_grid_y,
            hash_grid_z,
            cell_size,
            0.0,  # domain_min_x
            0.0,  # domain_min_y
            0.0,  # domain_min_z
        )
        
        # Phase 2: Compute prefix sum (cell_starts[i] = sum(counts[0:i]))
        self.ops.spatial_hash_prefix_sum(
            cell_counts,
            cell_starts,
            num_cells,
        )
        
        # Phase 3: Scatter particles to sorted array
        # First copy cell_starts to cell_offsets (working copy)
        cell_offsets = cell_starts[:num_cells].clone()
        self.ops.spatial_hash_scatter(
            particle_cell_idx,
            sorted_particle_idx,
            cell_offsets,
            n,
        )
        
        # Phase 4: Collision detection using spatial hash
        self.ops.spatial_hash_collisions(
            positions,
            velocities,
            excitations,
            masses,
            heats,
            sorted_particle_idx,
            cell_starts,
            particle_cell_idx,
            hash_grid_x,
            hash_grid_y,
            hash_grid_z,
            cell_size,
            0.0,  # domain_min_x
            0.0,  # domain_min_y
            0.0,  # domain_min_z
            float(cfg.dt),
            float(cfg.particle_radius),
            float(cfg.young_modulus),
            float(cfg.thermal_conductivity),
            float(cfg.restitution),
        )
        
        return velocities, excitations, heats
    
    def step(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
        use_spatial_hash: Optional[bool] = None,
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        """Execute one full physics timestep.
        
        Args:
            use_spatial_hash: If True, use O(N) spatial hash collisions.
                              If False, use O(N²) brute force.
                              If None (default), auto-select based on particle count:
                              - N < 1000: use brute force (lower overhead)
                              - N >= 1000: use spatial hash (better scaling)
        """
        # 1. Scatter particles to fields
        self.scatter_particles(positions, masses, heats)
        
        # 2. Solve field equations
        self.solve_gravity()
        self.diffuse_heat()
        
        # 3. Gather from fields and update particle state
        positions, velocities, energies, heats, excitations = self.gather_update_particles(
            positions, velocities, energies, heats, excitations, masses
        )
        
        # 4. Particle-particle interactions (collision + excitation transfer)
        # Auto-select collision algorithm based on particle count
        n = positions.size(0)
        if use_spatial_hash is None:
            use_spatial_hash = n >= 1000
        
        if use_spatial_hash:
            # O(N) spatial hash - better for large particle counts
            velocities, excitations, heats = self.compute_interactions_spatial_hash(
                positions, velocities, excitations, masses, heats
            )
        else:
            # O(N²) brute force - lower overhead for small counts
            velocities, excitations, heats = self.compute_interactions(
                positions, velocities, excitations, masses, heats
            )
        
        return positions, velocities, energies, heats, excitations


class SpectralCarrierPhysics:
    """Metal-accelerated spectral carriers (resonance potential, Langevin flow).

    This is the wave-space entanglement layer:
    - Oscillators are particles viewed as (phase θ, omega ω, amplitude A).
    - Carriers are global complex modes C_k with (omega_k, gate_width σ_k).
    - Updates are gradient flow on a resonance potential + Langevin noise.
    - Persistent *phase incoherence* triggers carrier splitting/spawning.
    """

    def __init__(
        self,
        config: SpectralCarrierConfig,
        grid_size: tuple[int, int, int],
        dt: float,
        device: str = "mps",
    ):
        if device != "mps":
            raise RuntimeError(f"SpectralCarrierPhysics requires device='mps', got '{device}'")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")

        self.config = config
        self.grid_size = grid_size
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.max_carriers = int(config.max_carriers)
        if self.max_carriers <= 0:
            raise ValueError("SpectralCarrierConfig.max_carriers must be > 0")

        if int(config.anchor_slots) != 8:
            raise ValueError(
                "SpectralCarrierConfig.anchor_slots must be 8 "
                "(must match CARRIER_ANCHORS in optimizer/metal/manifold_physics.metal)"
            )

        # Carrier state buffers (fixed capacity; active count tracked separately)
        self.carrier_real = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_imag = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_omega = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_gate_width = torch.full(
            (self.max_carriers,), float(config.gate_width_init), device=self.device, dtype=self.dtype
        )
        self.carrier_conflict = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.spawned_from_osc = torch.full(
            (self.max_carriers,), -1, device=self.device, dtype=torch.int32
        )

        # Memory state (GPU-owned)
        # carrier_state: 0=volatile, 1=stable, 2=crystallized
        self.carrier_state = torch.zeros(self.max_carriers, device=self.device, dtype=torch.int32)
        self.carrier_age = torch.zeros(self.max_carriers, device=self.device, dtype=torch.int32)

        # Anchors (flattened: max_carriers * anchor_slots)
        anchors = int(config.anchor_slots)
        self.anchor_slots = anchors
        self.carrier_anchor_idx = torch.full(
            (self.max_carriers * anchors,), -1, device=self.device, dtype=torch.int32
        )
        self.carrier_anchor_phase = torch.zeros(
            (self.max_carriers * anchors,), device=self.device, dtype=self.dtype
        )
        self.carrier_anchor_weight = torch.zeros(
            (self.max_carriers * anchors,), device=self.device, dtype=self.dtype
        )

        # Atomic counter buffer used by the Metal kernel (int32 backing storage)
        self._num_carriers_buf = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.num_carriers = 0

        # Random phases used when spawning carriers inside the kernel
        self._random_phases = torch.rand(self.max_carriers, device=self.device, dtype=self.dtype)

        # Global energy statistics buffer (GPU-written each step):
        # [mean_abs, mean, std, count]
        self._energy_stats = torch.zeros(4, device=self.device, dtype=self.dtype)

        # RNG seed for Langevin noise (host-controlled)
        self._rng_seed: int = 1

        # Load Metal ops
        from .jit import load_caramba_metal_ops
        self._ops = load_caramba_metal_ops()

    @property
    def ops(self):
        return self._ops

    def _ensure_seeded(self, osc_phase: "Tensor", osc_omega: "Tensor", osc_amp: "Tensor") -> None:
        """Genesis event: create the first carrier from the first oscillator.
        
        All other oscillators will couple to this carrier since it's the only one.
        Spectral conflict will build and trigger splits, creating new carriers
        that oscillators can then redistribute to based on frequency alignment.
        """
        if self.num_carriers > 0:
            return
        N = int(osc_phase.shape[0])
        if N == 0:
            return

        # Genesis: exactly ONE carrier from the first oscillator
        # (In a real system, this is the first particle that enters)
        idx = 0
        phi = osc_phase[idx].item()
        omega = osc_omega[idx].item()
        amp = osc_amp[idx].item()

        # Initialize carrier from the first oscillator phasor: C = z (no polarity hack)
        self.carrier_real[0] = amp * math.cos(phi)
        self.carrier_imag[0] = amp * math.sin(phi)
        self.carrier_omega[0] = omega
        # Gate width derived from wavelength: wider frequency = narrower pulse
        # For now, use initial value; this could be omega-dependent
        self.carrier_gate_width[0] = float(self.config.gate_width_init)
        self.carrier_conflict[0] = 0.0
        self.spawned_from_osc[0] = 0
        self.carrier_state[0] = 0
        self.carrier_age[0] = 0

        # Seed anchors: self-anchor the genesis oscillator.
        self.carrier_anchor_idx.fill_(-1)
        self.carrier_anchor_phase.zero_()
        self.carrier_anchor_weight.zero_()
        self.carrier_anchor_idx[0] = 0
        self.carrier_anchor_phase[0] = 0.0
        self.carrier_anchor_weight[0] = float(osc_amp[idx].item())

        self.num_carriers = 1
        self._num_carriers_buf[0] = 1

    def idle_compute(
        self,
        osc_phase: "Tensor",
        particle_excitations: "Tensor",
        particle_energies: "Tensor",
        *,
        steps: int = 1,
        mode: str = "explore",
    ) -> Dict[str, "Tensor"]:
        """Idle-time compute for spectral memory (runs on GPU).

        This is the Metal version of the old "idle pondering" ideas, renamed:
        - **consolidate**: stabilize and crystallize carriers (low noise)
        - **disambiguate**: reduce mode collisions (ω-space repulsion)
        - **explore**: counterfactual exploration with stochasticity + luck for weak links
        """
        mode_s = str(mode).lower().strip()
        if mode_s in ("consolidate", "consolidation", "stabilize"):
            m = 1
            temp = float(self.config.temperature) * 0.25
            anchor_eps = float(self.config.anchor_random_eps) * 0.25
            rand_energy_eps = float(self.config.topdown_random_energy_eps) * 0.25
            offender_floor = float(self.config.offender_weight_floor)
            repulsion = 0.0
        elif mode_s in ("disambiguate", "resolve", "separate"):
            m = 2
            temp = float(self.config.temperature) * 0.50
            anchor_eps = float(self.config.anchor_random_eps) * 0.50
            rand_energy_eps = float(self.config.topdown_random_energy_eps) * 0.50
            offender_floor = float(self.config.offender_weight_floor)
            repulsion = float(self.config.repulsion_scale)
        elif mode_s in ("explore", "counterfactual", "counterfactual_exploration"):
            m = 3
            temp = float(self.config.temperature) * 2.0
            # "weak bonds get lucky": raise ε and lower the weight floor
            anchor_eps = max(float(self.config.anchor_random_eps), 0.20)
            rand_energy_eps = max(float(self.config.topdown_random_energy_eps), 0.10)
            offender_floor = float(self.config.offender_weight_floor) * 0.10
            repulsion = 0.0
        else:
            raise ValueError(f"Unknown idle_compute mode: {mode!r}")

        out: Dict[str, "Tensor"] = {}
        for _ in range(int(steps)):
            osc_phase = osc_phase.contiguous()
            osc_omega = particle_excitations.to(dtype=self.dtype).contiguous()
            energy = particle_energies.to(dtype=self.dtype).contiguous()
            osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

            self.ops.thermo_reduce_energy_stats(energy, self._energy_stats)
            self._ensure_seeded(osc_phase, osc_omega, osc_amp)
            if self.num_carriers == 0:
                out = {
                    "frequencies": torch.empty(0, device=self.device, dtype=self.dtype),
                    "gate_widths": torch.empty(0, device=self.device, dtype=self.dtype),
                    "amplitudes": torch.empty(0, device=self.device, dtype=self.dtype),
                    "phases": torch.empty(0, device=self.device, dtype=self.dtype),
                    "conflict": torch.empty(0, device=self.device, dtype=self.dtype),
                    "osc_phase": osc_phase,
                    "osc_energy": energy,
                    "carrier_state": torch.empty(0, device=self.device, dtype=torch.int32),
                    "carrier_age": torch.empty(0, device=self.device, dtype=torch.int32),
                }
                break

            self._num_carriers_buf[0] = int(self.num_carriers)
            self._random_phases.uniform_()
            self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
            cfg = self.config

            self.ops.spectral_carrier_update_and_split(
                osc_phase,
                osc_omega,
                osc_amp,
                self.carrier_real,
                self.carrier_imag,
                self.carrier_omega,
                self.carrier_gate_width,
                self.carrier_conflict,
                self.carrier_state,
                self.carrier_age,
                self.carrier_anchor_idx,
                self.carrier_anchor_phase,
                self.carrier_anchor_weight,
                self._num_carriers_buf,
                self.spawned_from_osc,
                self._random_phases,
                self._energy_stats,
                int(self.num_carriers),
                int(self.max_carriers),
                float(self.dt),
                float(cfg.coupling_scale),
                float(cfg.carrier_reg),
                float(temp),
                int(self._rng_seed) & 0xFFFFFFFF,
                float(cfg.conflict_threshold),
                float(offender_floor),
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
                float(cfg.ema_alpha),
                float(cfg.recenter_alpha),
                int(m),
                float(anchor_eps),
                float(cfg.stable_amp_threshold),
                float(cfg.crystallize_amp_threshold),
                float(cfg.crystallize_conflict_threshold),
                int(cfg.crystallize_age),
                float(cfg.crystallized_coupling_boost),
                float(cfg.volatile_decay_mul),
                float(cfg.stable_decay_mul),
                float(cfg.crystallized_decay_mul),
                float(cfg.topdown_phase_scale),
                float(cfg.topdown_energy_scale),
                float(rand_energy_eps),
                float(repulsion),
            )

            new_count = int(self._num_carriers_buf.to("cpu").item())
            self.num_carriers = max(0, min(new_count, self.max_carriers))

            self.ops.spectral_topdown_bias_energies(
                energy,
                osc_amp,
                self.carrier_state,
                self.carrier_anchor_idx,
                self.carrier_anchor_weight,
                int(self.num_carriers),
                int(self.max_carriers),
                float(self.dt),
                int(self._rng_seed) & 0xFFFFFFFF,
                int(m),
                float(cfg.topdown_energy_scale),
                float(rand_energy_eps),
            )

            osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()
            self.ops.spectral_update_oscillator_phases(
                osc_phase,
                osc_omega,
                osc_amp,
                self.carrier_real,
                self.carrier_imag,
                self.carrier_omega,
                self.carrier_gate_width,
                self.carrier_state,
                self.carrier_anchor_idx,
                self.carrier_anchor_phase,
                self.carrier_anchor_weight,
                self._energy_stats,
                int(self.num_carriers),
                int(self.max_carriers),
                float(self.dt),
                float(cfg.coupling_scale),
                float(temp),
                int(self._rng_seed) & 0xFFFFFFFF,
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
                float(cfg.crystallized_coupling_boost),
                float(cfg.topdown_phase_scale),
            )

            self._num_carriers_buf[0] = int(self.num_carriers)
            new_count = self.ops.spectral_spawn_uncoupled(
                osc_phase,
                osc_omega,
                osc_amp,
                self.carrier_omega,
                self.carrier_gate_width,
                self.carrier_real,
                self.carrier_imag,
                self.carrier_omega,
                self.carrier_gate_width,
                self.carrier_conflict,
                self.carrier_state,
                self.carrier_age,
                self.carrier_anchor_idx,
                self.carrier_anchor_phase,
                self.carrier_anchor_weight,
                self._num_carriers_buf,
                int(self.num_carriers),
                int(self.max_carriers),
                float(cfg.uncoupled_threshold),
                float(cfg.gate_width_init),
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
            )
            self.num_carriers = int(new_count)

            cr = self.carrier_real[: self.num_carriers]
            ci = self.carrier_imag[: self.num_carriers]
            amp = torch.sqrt(cr * cr + ci * ci)
            phase = torch.atan2(ci, cr)
            out = {
                "frequencies": self.carrier_omega[: self.num_carriers],
                "gate_widths": self.carrier_gate_width[: self.num_carriers],
                "amplitudes": amp,
                "phases": phase,
                "conflict": self.carrier_conflict[: self.num_carriers],
                "osc_phase": osc_phase,
                "osc_energy": energy,
                "carrier_state": self.carrier_state[: self.num_carriers],
                "carrier_age": self.carrier_age[: self.num_carriers],
            }
            particle_energies = energy
        return out

    def step(
        self,
        osc_phase: "Tensor",           # (N,) fp32 MPS in/out
        particle_excitations: "Tensor",# (N,) fp32 MPS
        particle_energies: "Tensor",   # (N,) fp32 MPS
    ) -> Dict[str, "Tensor"]:
        """One spectral carrier tick (update + split + oscillator phase update)."""
        osc_phase = osc_phase.contiguous()
        osc_omega = particle_excitations.to(dtype=self.dtype).contiguous()
        energy = particle_energies.to(dtype=self.dtype).contiguous()
        osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

        # Adaptive renormalization: compute global energy statistics on-GPU.
        self.ops.thermo_reduce_energy_stats(energy, self._energy_stats)

        self._ensure_seeded(osc_phase, osc_omega, osc_amp)
        if self.num_carriers == 0:
            return {
                "frequencies": torch.empty(0, device=self.device, dtype=self.dtype),
                "gate_widths": torch.empty(0, device=self.device, dtype=self.dtype),
                "amplitudes": torch.empty(0, device=self.device, dtype=self.dtype),
                "phases": torch.empty(0, device=self.device, dtype=self.dtype),
                "conflict": torch.empty(0, device=self.device, dtype=self.dtype),
                "osc_phase": osc_phase,
            }

        # Ensure counter starts at current carrier count (kernel increments it on split)
        self._num_carriers_buf[0] = int(self.num_carriers)
        self._random_phases.uniform_()

        cfg = self.config
        # Advance RNG seed so noise changes each step deterministically
        self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
        self.ops.spectral_carrier_update_and_split(
            osc_phase,
            osc_omega,
            osc_amp,
            self.carrier_real,
            self.carrier_imag,
            self.carrier_omega,
            self.carrier_gate_width,
            self.carrier_conflict,
            self.carrier_state,
            self.carrier_age,
            self.carrier_anchor_idx,
            self.carrier_anchor_phase,
            self.carrier_anchor_weight,
            self._num_carriers_buf,
            self.spawned_from_osc,
            self._random_phases,
            self._energy_stats,
            int(self.num_carriers),
            int(self.max_carriers),
            float(self.dt),
            float(cfg.coupling_scale),
            float(cfg.carrier_reg),
            float(cfg.temperature),
            int(self._rng_seed) & 0xFFFFFFFF,
            float(cfg.conflict_threshold),
            float(cfg.offender_weight_floor),
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
            float(cfg.ema_alpha),
            float(cfg.recenter_alpha),
            0,  # mode (0=online)
            float(cfg.anchor_random_eps),
            float(cfg.stable_amp_threshold),
            float(cfg.crystallize_amp_threshold),
            float(cfg.crystallize_conflict_threshold),
            int(cfg.crystallize_age),
            float(cfg.crystallized_coupling_boost),
            float(cfg.volatile_decay_mul),
            float(cfg.stable_decay_mul),
            float(cfg.crystallized_decay_mul),
            float(cfg.topdown_phase_scale),
            float(cfg.topdown_energy_scale),
            float(cfg.topdown_random_energy_eps),
            float(cfg.repulsion_scale),
        )

        # Read updated carrier count (sync via CPU copy)
        new_count = int(self._num_carriers_buf.to("cpu").item())
        self.num_carriers = max(0, min(new_count, self.max_carriers))

        # Top-down energy bias (crystallized carriers act as priors/completions).
        # NOTE: This updates `energy` (a local fp32 view); callers can choose to
        # use the returned "osc_energy" for downstream inference.
        self.ops.spectral_topdown_bias_energies(
            energy,
            osc_amp,
            self.carrier_state,
            self.carrier_anchor_idx,
            self.carrier_anchor_weight,
            int(self.num_carriers),
            int(self.max_carriers),
            float(self.dt),
            int(self._rng_seed) & 0xFFFFFFFF,
            0,  # mode (0=online)
            float(cfg.topdown_energy_scale),
            float(cfg.topdown_random_energy_eps),
        )

        # Recompute amplitude after energy injection so phase update sees the bias.
        osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

        # Oscillator phase update from carriers
        self.ops.spectral_update_oscillator_phases(
            osc_phase,
            osc_omega,
            osc_amp,
            self.carrier_real,
            self.carrier_imag,
            self.carrier_omega,
            self.carrier_gate_width,
            self.carrier_state,
            self.carrier_anchor_idx,
            self.carrier_anchor_phase,
            self.carrier_anchor_weight,
            self._energy_stats,
            int(self.num_carriers),
            int(self.max_carriers),
            float(self.dt),
            float(cfg.coupling_scale),
            float(cfg.temperature),
            int(self._rng_seed) & 0xFFFFFFFF,
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
            float(cfg.crystallized_coupling_boost),
            float(cfg.topdown_phase_scale),
        )

        # Spawn carriers for any uncoupled oscillators
        # (ensures every oscillator is coupled to at least one carrier)
        self._num_carriers_buf[0] = int(self.num_carriers)
        new_count = self.ops.spectral_spawn_uncoupled(
            osc_phase,
            osc_omega,
            osc_amp,
            self.carrier_omega,
            self.carrier_gate_width,
            self.carrier_real,
            self.carrier_imag,
            self.carrier_omega,      # write to same buffer
            self.carrier_gate_width, # write to same buffer
            self.carrier_conflict,
            self.carrier_state,
            self.carrier_age,
            self.carrier_anchor_idx,
            self.carrier_anchor_phase,
            self.carrier_anchor_weight,
            self._num_carriers_buf,
            int(self.num_carriers),
            int(self.max_carriers),
            float(cfg.uncoupled_threshold),  # oscillators with total coupling < this spawn
            float(cfg.gate_width_init),
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
        )
        self.num_carriers = int(new_count)

        # Prepare state views for dashboard
        cr = self.carrier_real[: self.num_carriers]
        ci = self.carrier_imag[: self.num_carriers]
        amp = torch.sqrt(cr * cr + ci * ci)
        phase = torch.atan2(ci, cr)

        return {
            "frequencies": self.carrier_omega[: self.num_carriers],
            "gate_widths": self.carrier_gate_width[: self.num_carriers],
            "amplitudes": amp,
            "phases": phase,
            "conflict": self.carrier_conflict[: self.num_carriers],
            "osc_phase": osc_phase,
            "osc_energy": energy,
            "carrier_state": self.carrier_state[: self.num_carriers],
            "carrier_age": self.carrier_age[: self.num_carriers],
        }


class ParticleGenerator:
    """Metal-accelerated particle generation for file injection."""
    
    # Pattern constants
    PATTERN_CLUSTER = 0
    PATTERN_LINE = 1
    PATTERN_SPHERE = 2
    PATTERN_RANDOM = 3
    PATTERN_GRID = 4
    
    PATTERN_NAMES = ["cluster", "line", "sphere", "random", "grid"]
    
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        device: str = "mps",
    ):
        if device != "mps":
            raise RuntimeError(f"ParticleGenerator requires device='mps', got '{device}'")
        
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.file_count = 0
        
        from .jit import load_caramba_metal_ops
        self._ops = load_caramba_metal_ops()
    
    @property
    def ops(self):
        return self._ops
    
    def generate_file(
        self,
        num_particles: int = 50,
        pattern: Optional[str] = None,
        energy_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a synthetic 'file' - a burst of related particles.
        
        All computation happens on GPU via Metal kernels.
        """
        import random
        
        self.file_count += 1
        
        if pattern is None:
            pattern = random.choice(self.PATTERN_NAMES)
        
        pattern_id = self.PATTERN_NAMES.index(pattern) if pattern in self.PATTERN_NAMES else 3
        
        gx, gy, gz = self.grid_size
        
        # Random center for cluster/sphere patterns
        center_x = random.uniform(gx * 0.2, gx * 0.8)
        center_y = random.uniform(gy * 0.2, gy * 0.8)
        center_z = random.uniform(gz * 0.2, gz * 0.8)
        
        # Spread depends on pattern
        grid_min = min(gx, gy, gz)
        if pattern == "cluster":
            spread = grid_min * 0.15
        elif pattern == "sphere":
            spread = grid_min * 0.2 + random.uniform(0, 5)
        elif pattern == "line":
            spread = grid_min * 0.7
        else:
            spread = grid_min * 0.4
        
        # Direction for line pattern
        dir_vec = torch.randn(3)
        dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
        
        # Allocate output tensors
        positions = torch.empty(num_particles, 3, device=self.device, dtype=self.dtype)
        velocities = torch.empty(num_particles, 3, device=self.device, dtype=self.dtype)
        energies = torch.empty(num_particles, device=self.device, dtype=self.dtype)
        heats = torch.empty(num_particles, device=self.device, dtype=self.dtype)
        excitations = torch.empty(num_particles, device=self.device, dtype=self.dtype)
        masses = torch.empty(num_particles, device=self.device, dtype=self.dtype)
        
        # Random values for generation
        random_pos = torch.rand(num_particles, 3, device=self.device, dtype=self.dtype)
        random_props = torch.rand(num_particles, 4, device=self.device, dtype=self.dtype)
        
        self.ops.generate_particles(
            positions,
            velocities,
            energies,
            heats,
            excitations,
            masses,
            random_pos,
            random_props,
            pattern_id,
            float(gx),
            float(gy),
            float(gz),
            energy_scale,
            center_x,
            center_y,
            center_z,
            spread,
            float(dir_vec[0]),
            float(dir_vec[1]),
            float(dir_vec[2]),
        )
        
        return {
            "positions": positions,
            "velocities": velocities,
            "energies": energies,
            "heats": heats,
            "excitations": excitations,
            "masses": masses,
            "pattern": pattern,
            "file_id": self.file_count,
        }
