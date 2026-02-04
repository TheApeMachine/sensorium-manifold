"""Metal manifold physics kernels.

Single implementation using Metal acceleration. No fallbacks.
If Metal is not available or something fails, we raise an exception.

------------------------------------------------------------------------------
COMMENT CONVENTION (physics choices)
------------------------------------------------------------------------------
When we choose a value, method, or equation form, we annotate it using a
structured block so the codebase stays auditable.

Format:
  # [CHOICE] <name>
  # [FORMULA] <math / equation / mapping>
  # [REASON] <brief why this form/value>
  # [NOTES] <brief caveats, assumptions, invariants, TODOs>

The intent is:
- correctness first: make the modeled physics explicit and falsifiable
- performance second: make optimizations explicit and non-semantic
------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch

from optimizer.physics_units import PhysicalConstants, UnitSystem, assert_finite_constants

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
    
    # Backend router requires this.
    device: str = "mps"

    grid_size: tuple[int, int, int] = (64, 64, 64)
    # Simulation parameters
    # [CHOICE] grid_spacing (Δx)
    # [FORMULA] world_length = grid_size * Δx
    # [REASON] sets the discretization length scale
    # [NOTES] this is not a “knob”; it defines the numerical resolution / units.
    grid_spacing: float = 1.0

    # [CHOICE] dt (Δt)
    # [FORMULA] integrator step size for particle update
    # [REASON] numerical integration time step
    # [NOTES] currently used as pseudo-time for relaxation; should become derived
    #         from explicit stability/accuracy requirements (not tuned).
    dt: float = 0.01
    
    # Adaptive step sizing (numerical stability, not "physics knobs").
    # If enabled, `ManifoldPhysics.step()` derives a safe dt from current state
    # and discretization scales, instead of relying on hard clamps in kernels.
    adaptive_dt: bool = False
    dt_min: float = 1e-4
    dt_max: float = 0.02

    # Diagnostics / instrumentation
    # If True, `ManifoldPhysics.step()` will compute an energy report using `.item()`
    # reductions (this synchronizes with the device). Keep False for performance.
    enable_energy_report: bool = False

    # Debugging (correctness-first; may force device sync)
    # If enabled, `ManifoldPhysics.step()` checks tensors for NaN/inf at key
    # sub-stages and raises immediately when detected.
    debug_check_finite: bool = False
    debug_check_finite_every: int = 1
    
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
    # [CHOICE] unit system (simulation units → SI units)
    # [FORMULA] x_SI = x_sim * unit_scale
    # [REASON] makes “universal constants” non-tuneable by deriving them from CODATA
    # [NOTES] Default is the identity mapping (1 sim unit == 1 SI unit).
    unit_system: UnitSystem = field(default_factory=UnitSystem.si)
    
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

    def validate_invariants(self) -> None:
        """Fail loudly on invalid physical/scalar invariants.

        [CHOICE] invariant validation (host-side, scalar)
        [FORMULA] require: Δx>0, Δt>0, r>0, c_v>0, κ>=0, μ>=0, E>=0, 0<=ε<=1
        [REASON] invalid parameters should not be “handled” by silent clamps in kernels
        [NOTES] tensor invariants (e.g. masses>0) are enforced kernel-side via NaN
                sentinels to avoid host synchronizations in hot paths.
        """
        if not (self.grid_spacing > 0.0):
            raise ValueError(f"grid_spacing must be > 0, got {self.grid_spacing}")
        if not (self.dt > 0.0):
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if not (self.particle_radius > 0.0):
            raise ValueError(f"particle_radius must be > 0, got {self.particle_radius}")
        if not (self.specific_heat > 0.0):
            raise ValueError(f"specific_heat must be > 0, got {self.specific_heat}")
        if not (self.thermal_conductivity >= 0.0):
            raise ValueError(f"thermal_conductivity must be >= 0, got {self.thermal_conductivity}")
        if not (self.dynamic_viscosity >= 0.0):
            raise ValueError(f"dynamic_viscosity must be >= 0, got {self.dynamic_viscosity}")
        if not (self.young_modulus >= 0.0):
            raise ValueError(f"young_modulus must be >= 0, got {self.young_modulus}")
        if not (0.0 <= self.emissivity <= 1.0):
            raise ValueError(f"emissivity must be in [0,1], got {self.emissivity}")
        if not (0.0 <= self.restitution <= 1.0):
            raise ValueError(f"restitution must be in [0,1], got {self.restitution}")

        # [CHOICE] unit system validity (scalar invariants)
        # [FORMULA] require base-unit scales > 0
        # [REASON] constants conversion divides by powers of these scales
        us = self.unit_system
        if not (us.length_unit_m > 0.0 and us.mass_unit_kg > 0.0 and us.time_unit_s > 0.0 and us.temperature_unit_K > 0.0):
            raise ValueError(f"unit_system base scales must be > 0, got {us!r}")

    def physical_constants(self) -> PhysicalConstants:
        """Return physical constants expressed in simulation units.

        [CHOICE] constants source
        [FORMULA] CODATA(SI) → sim via base-unit exponents (UnitSystem)
        [REASON] make the source of constants auditable and explicit
        [NOTES] Universal values are derived from CODATA + UnitSystem (no legacy knobs).
        """
        c = PhysicalConstants.from_codata_si(self.unit_system)
        assert_finite_constants(c)
        return c


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

    def validate_invariants(self) -> None:
        """Fail loudly on invalid scalar invariants for the spectral layer."""
        if not (self.max_carriers > 0):
            raise ValueError(f"max_carriers must be > 0, got {self.max_carriers}")
        if not (self.gate_width_init > 0.0):
            raise ValueError(f"gate_width_init must be > 0, got {self.gate_width_init}")
        if not (self.gate_width_min > 0.0):
            raise ValueError(f"gate_width_min must be > 0, got {self.gate_width_min}")
        if not (self.gate_width_max >= self.gate_width_min):
            raise ValueError(f"gate_width_max must be >= gate_width_min, got {self.gate_width_max} < {self.gate_width_min}")
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0,1], got {self.ema_alpha}")
        if not (0.0 <= self.recenter_alpha <= 1.0):
            raise ValueError(f"recenter_alpha must be in [0,1], got {self.recenter_alpha}")
        if not (self.anchor_slots == 8):
            raise ValueError("anchor_slots must be 8 (must match CARRIER_ANCHORS in Metal)")


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
        # NOTE: `gravity_field` is actually the scattered mass-per-cell field (ρ_mass * cell_volume).
        # We keep the historical name for compatibility.
        self.gravity_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        # Scattered heat-per-cell field Q_cell (same units as particle heat).
        self.heat_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.gravity_potential = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        # True temperature field T_cell derived from (Q_cell / (m_cell * c_v)).
        self.temperature_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        
        # Load Metal ops immediately (fail fast)
        from .jit import load_caramba_metal_ops
        self._ops = load_caramba_metal_ops()

        # Diagnostics (optional, lightweight)
        self.last_dt: float = float(self.config.dt)
        self.last_energy_report: Optional[dict[str, float]] = None

        # FFT Poisson cache (periodic domain).
        self._fft_k2: Optional[torch.Tensor] = None      # (gx, gy, gz) float32 on MPS
        self._fft_inv_k2: Optional[torch.Tensor] = None  # (gx, gy, gz) float32 on MPS

        # Collision snapshots (avoid write hazards by reading old state).
        self._vel_in: Optional[torch.Tensor] = None   # (N, 3)
        self._heat_in: Optional[torch.Tensor] = None  # (N,)

        # Spatial-hash scratch buffers (reused to avoid per-step allocations).
        # Shapes depend on N and on derived hash grid dims (num_cells).
        self._hash_particle_cell_idx: Optional[torch.Tensor] = None  # (N,) int32
        self._hash_sorted_particle_idx: Optional[torch.Tensor] = None  # (N,) int32
        self._hash_cell_counts: Optional[torch.Tensor] = None  # (num_cells,) int32 (must be zeroed each call)
        self._hash_cell_starts: Optional[torch.Tensor] = None  # (num_cells+1,) int32
        self._hash_cell_offsets: Optional[torch.Tensor] = None  # (num_cells,) int32 (working copy of starts)

    def synchronize(self) -> None:
        """Force completion of queued GPU work (execution barrier).

        [CHOICE] explicit MPS synchronization hook
        [FORMULA] torch.mps.synchronize()
        [REASON] MPS execution is asynchronous; observers may otherwise sample
                 stale/partially-updated tensors when mixing device/host work.
        [NOTES] Use sparingly; forcing the queue to drain can reduce throughput.
        """
        if self.device.type == "mps":
            torch.mps.synchronize()

    def done_thinking(
        self,
        velocities: "Tensor",
        masses: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        dt: Optional[float] = None,
    ) -> bool:
        """Return True when the spatial system is sufficiently settled.

        This is intended for ``relax until equilibrium'' loops. It is
        correctness-first and intentionally derived from numerical resolution
        (no hand-tuned thresholds).

        [CHOICE] settledness test (numerically-derived)
        [FORMULA]
          tol = sqrt(eps_fp32)
          (1) max(||v||) * Δt <= tol * Δx
          (2) Σ E_kin <= tol * Σ (Q + E_osc)
        [REASON]
          - (1) means particles move less than a floating-precision fraction of a cell per step.
          - (2) means directed motion is negligible compared to internal energy.
        [NOTES]
          - Uses reductions + `.item()` (device sync). Call sparingly (e.g. every N steps).
          - If there are no particles, returns True.
        """
        if velocities.numel() == 0:
            return True

        # Δt: prefer caller-provided; fall back to last derived dt.
        dt_eff = float(self.last_dt if dt is None else dt)
        dx = float(self.config.grid_spacing)

        # Numerically-derived tolerance (no knobs).
        tol = float(math.sqrt(torch.finfo(torch.float32).eps))

        v = velocities.to(torch.float32)
        m = masses.to(torch.float32)
        q = heats.to(torch.float32)
        e = energies.to(torch.float32)

        # (1) displacement per step small relative to grid resolution.
        max_speed = float(torch.linalg.vector_norm(v, dim=1).max().detach().item())
        if not (max_speed * dt_eff <= tol * dx):
            return False

        # (2) kinetic energy negligible relative to internal energy.
        v2 = (v * v).sum(dim=1)
        ke_sum = float((0.5 * m * v2).sum().detach().item())
        u_sum = float((q + e).sum().detach().item())
        if u_sum <= 0.0:
            return ke_sum == 0.0
        return ke_sum <= (tol * u_sum)

    def _ensure_spatial_hash_buffers(self, n: int, num_cells: int) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        """Allocate/reuse spatial-hash scratch buffers on the correct device.

        This avoids repeated allocations and clones in the hot path.
        """
        dev = self.device
        if (
            self._hash_particle_cell_idx is None
            or self._hash_particle_cell_idx.numel() != n
            or self._hash_particle_cell_idx.device != dev
        ):
            self._hash_particle_cell_idx = torch.empty(n, dtype=torch.int32, device=dev)
        if (
            self._hash_sorted_particle_idx is None
            or self._hash_sorted_particle_idx.numel() != n
            or self._hash_sorted_particle_idx.device != dev
        ):
            self._hash_sorted_particle_idx = torch.empty(n, dtype=torch.int32, device=dev)
        if (
            self._hash_cell_counts is None
            or self._hash_cell_counts.numel() != num_cells
            or self._hash_cell_counts.device != dev
        ):
            self._hash_cell_counts = torch.empty(num_cells, dtype=torch.int32, device=dev)
        if (
            self._hash_cell_starts is None
            or self._hash_cell_starts.numel() != (num_cells + 1)
            or self._hash_cell_starts.device != dev
        ):
            self._hash_cell_starts = torch.empty(num_cells + 1, dtype=torch.int32, device=dev)
        if (
            self._hash_cell_offsets is None
            or self._hash_cell_offsets.numel() != num_cells
            or self._hash_cell_offsets.device != dev
        ):
            self._hash_cell_offsets = torch.empty(num_cells, dtype=torch.int32, device=dev)

        return (
            self._hash_particle_cell_idx,
            self._hash_cell_counts,
            self._hash_cell_starts,
            self._hash_sorted_particle_idx,
            self._hash_cell_offsets,
        )

    def _ensure_collision_snapshots(self, velocities: "Tensor", heats: "Tensor") -> tuple["Tensor", "Tensor"]:
        """Allocate/reuse snapshot buffers and copy current state into them."""
        if self._vel_in is None or self._vel_in.shape != velocities.shape or self._vel_in.device != velocities.device:
            self._vel_in = torch.empty_like(velocities)
        if self._heat_in is None or self._heat_in.shape != heats.shape or self._heat_in.device != heats.device:
            self._heat_in = torch.empty_like(heats)
        self._vel_in.copy_(velocities)
        self._heat_in.copy_(heats)
        return self._vel_in, self._heat_in

    def _ensure_fft_poisson_cache(self) -> None:
        """Precompute 1/k^2 grid for FFT Poisson on the current device."""
        if self._fft_inv_k2 is not None:
            return
        gx, gy, gz = self.grid_dims
        h = float(self.config.grid_spacing)
        two_pi = float(2.0 * math.pi)

        # Wave numbers in radians per unit length.
        kx = two_pi * torch.fft.fftfreq(gx, d=h, device=self.device, dtype=torch.float32)
        ky = two_pi * torch.fft.fftfreq(gy, d=h, device=self.device, dtype=torch.float32)
        kz = two_pi * torch.fft.fftfreq(gz, d=h, device=self.device, dtype=torch.float32)

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = (KX * KX + KY * KY + KZ * KZ).to(torch.float32)
        inv_k2 = torch.zeros_like(k2)
        inv_k2 = torch.where(k2 > 0, 1.0 / k2, inv_k2)  # k=0 -> 0 (gauge)
        self._fft_k2 = k2
        self._fft_inv_k2 = inv_k2
    
    @property
    def ops(self):
        return self._ops
    
    def scatter_particles(
        self,
        positions: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
        energies: Optional["Tensor"] = None,
    ) -> None:
        """Scatter particle mass and total internal energy to grid, then derive temperature."""
        self.ops.manifold_clear_field(self.gravity_field)
        self.ops.manifold_clear_field(self.heat_field)
        pos = positions.contiguous()
        m = masses.contiguous()
        q = heats.contiguous()
        e = (energies if energies is not None else torch.zeros_like(q)).contiguous()

        # [CHOICE] particle→grid scatter implementation (performance-only)
        # [FORMULA] identical deposition (trilinear P2G); differs only in accumulation order
        # [REASON] tiled scatter reduces global atomic contention at large N
        # [NOTES] For very small N, the simple atomic scatter can be faster.
        if pos.numel() >= 4096 * 3:
            gx, gy, gz = self.grid_dims
            self.ops.scatter_particle_tiled(
                pos,
                m,
                q,
                e,
                self.gravity_field,
                self.heat_field,
                int(gx),
                int(gy),
                int(gz),
                float(self.config.grid_spacing),
            )
        else:
            self.ops.manifold_scatter_particles(
                pos,
                m,
                q,
                e,
                self.gravity_field,
                self.heat_field,
                float(self.config.grid_spacing),
            )

        # [CHOICE] temperature field derivation (cell-level)
        # [FORMULA] If m_cell > 0: T_cell = Q_cell / (m_cell * c_v); else T_cell = 0
        # [REASON] defines local thermodynamic state from conserved per-cell stores
        # [NOTES] Vacuum semantics are explicit: empty cells have T=0.
        self.ops.manifold_derive_temperature(
            self.gravity_field,
            self.heat_field,
            self.temperature_field,
            float(self.config.specific_heat),
        )
    
    def solve_gravity(self) -> None:
        """Solve Poisson equation for gravitational potential.
        
        Poisson equation: ∇²φ = 4πGρ
        We use G from the config as the gravitational constant.
        """
        cfg = self.config
        const = cfg.physical_constants()
        # [CHOICE] gravity field solve (periodic Poisson via FFT)
        # [FORMULA] ∇²φ = 4πGρ
        #          φ̂(k) = -(4πG/k²) ρ̂(k), with φ̂(0)=0
        # [REASON] exact global solve for periodic domains; avoids slow Jacobi iteration
        # [NOTES] gauge choice: subtract mean(ρ) so k=0 mode is zero (well-defined).
        #         IMPORTANT: if φ includes G via the Poisson solve, then the particle
        #         acceleration should be a = -∇φ (no additional G factor later).
        #         We currently multiply by G again in the Metal force kernel → likely
        #         double-counting. This must be fixed for correctness.
        self._ensure_fft_poisson_cache()
        assert self._fft_inv_k2 is not None
        h = float(cfg.grid_spacing)
        rho = self.gravity_field.to(torch.float32) / (h * h * h)  # mass-per-cell -> density
        rho = rho - rho.mean()
        rho_hat = torch.fft.fftn(rho)
        phi_hat = -(4.0 * math.pi * float(const.G)) * rho_hat * self._fft_inv_k2
        phi = torch.fft.ifftn(phi_hat).real.to(self.dtype)
        self.gravity_potential.copy_(phi)
    
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

    def _derive_dt(
        self,
        velocities: "Tensor",
        masses: "Tensor",
    ) -> float:
        """Derive a stable step size from current state.
        
        This is *pseudo-time* (relaxation step). The goal is numerical stability
        without kernel-side clamps.
        """
        cfg = self.config
        if not bool(getattr(cfg, "adaptive_dt", False)):
            return float(cfg.dt)
        # [CHOICE] adaptive dt (disabled: would require host sync)
        # [FORMULA] N/A
        # [REASON] the current implementation needs `.item()` reductions (CPU sync)
        # [NOTES] TODO: implement device-side dt computation + pass dt via device buffer
        #         to all kernels. Until then, we fail loudly if enabled.
        raise RuntimeError(
            "adaptive_dt=True is currently disabled because it requires host synchronization. "
            "Set adaptive_dt=False and choose dt via unit system / discretization, or implement "
            "a device-side dt pipeline."
        )
    
    def gather_update_particles(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
        *,
        dt: Optional[float] = None,
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        """Gather from fields and update all particle state.
        
        Uses fundamental physics:
        - Gravity: F = -m ∇φ   (with G included in φ via the Poisson solve)
        - Pressure: a = -(1/ρ)∇P, with EOS P = ρ k_B T (vacuum: ρ<=0 ⇒ pressure=0)
        - Heat transfer: Newton's law + Stefan-Boltzmann radiation
        - Drag: Stokes' law F = -6πηrv
        """
        cfg = self.config
        const = cfg.physical_constants()
        if dt is None:
            dt = float(cfg.dt)
        
        positions = positions.contiguous()
        velocities = velocities.contiguous()
        energies = energies.contiguous()
        heats = heats.contiguous()
        excitations = excitations.contiguous()
        masses = masses.contiguous()
        
        self.ops.manifold_gather_update_particles(
            self.gravity_potential,
            self.temperature_field,
            self.gravity_field,
            positions,
            velocities,
            energies,
            heats,
            excitations,
            masses,
            float(dt),
            float(cfg.grid_spacing),
            # Fundamental constants
            float(const.G),
            float(const.k_B),
            float(const.sigma_SB),
            float(const.hbar),
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
        *,
        dt: Optional[float] = None,
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
        if dt is None:
            dt = float(cfg.dt)
        
        positions = positions.contiguous()
        velocities = velocities.contiguous()
        excitations = excitations.contiguous()
        masses = masses.contiguous()
        heats = heats.contiguous()
        
        vel_in, heat_in = self._ensure_collision_snapshots(velocities, heats)
        self.ops.particle_interactions(
            positions,
            velocities,
            excitations,
            masses,
            heats,
            vel_in,
            heat_in,
            float(dt),
            float(cfg.particle_radius),
            float(cfg.young_modulus),
            float(cfg.thermal_conductivity),
            float(cfg.specific_heat),
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
        *,
        dt: Optional[float] = None,
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
        if dt is None:
            dt = float(cfg.dt)
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
        
        # Allocate/reuse working buffers (hot path).
        particle_cell_idx, cell_counts, cell_starts, sorted_particle_idx, cell_offsets = (
            self._ensure_spatial_hash_buffers(n, num_cells)
        )
        # IMPORTANT: cell_counts is used with atomics in the kernel and must be zeroed.
        cell_counts.zero_()
        
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
        
        # Phase 2: Compute prefix sum (cell_starts[i] = sum(counts[0:i])).
        # [CHOICE] GPU exclusive scan (no CPU sync; scalable)
        # [FORMULA] cell_starts[i] = Σ_{j<i} cell_counts[j]
        # [REASON] enables O(N) spatial hashing without single-thread bottleneck
        # [NOTES] Implemented via hierarchical block scans; uses uint32 semantics in int32 tensors.
        self._exclusive_scan_u32_into_starts(cell_counts, cell_starts, num_cells)
        
        # Phase 3: Scatter particles to sorted array
        # Copy starts to offsets (working copy) without allocating.
        cell_offsets.copy_(cell_starts[:num_cells])
        self.ops.spatial_hash_scatter(
            particle_cell_idx,
            sorted_particle_idx,
            cell_offsets,
            n,
        )
        
        # Phase 4: Collision detection using spatial hash
        vel_in, heat_in = self._ensure_collision_snapshots(velocities, heats)
        self.ops.spatial_hash_collisions(
            positions,
            velocities,
            excitations,
            masses,
            heats,
            sorted_particle_idx,
            cell_starts,
            particle_cell_idx,
            vel_in,
            heat_in,
            hash_grid_x,
            hash_grid_y,
            hash_grid_z,
            cell_size,
            0.0,  # domain_min_x
            0.0,  # domain_min_y
            0.0,  # domain_min_z
            float(dt),
            float(cfg.particle_radius),
            float(cfg.young_modulus),
            float(cfg.thermal_conductivity),
            float(cfg.specific_heat),
            float(cfg.restitution),
        )
        
        return velocities, excitations, heats

    def _exclusive_scan_u32_into_starts(self, counts_u32_i32: "Tensor", starts_u32_i32: "Tensor", n: int) -> None:
        """Exclusive scan of int32 tensor (uint32 semantics) into starts[:n] and starts[n]=total."""
        if n <= 0:
            starts_u32_i32.zero_()
            return

        # [CHOICE] hierarchical block scan (no host sync)
        # [FORMULA] exclusive_scan(x)[i] = Σ_{j<i} x[j]
        # [REASON] supports large n by scanning block sums recursively on GPU
        # [NOTES] buffers are int32 with uint32 semantics.
        tg = 256
        if not hasattr(self, "_scan_scratch"):
            self._scan_scratch = {}

        def _scratch(name: str, size: int) -> "Tensor":
            key = (name, int(size))
            t = self._scan_scratch.get(key)
            if t is None or t.numel() != size or t.device != counts_u32_i32.device:
                t = torch.empty((size,), device=counts_u32_i32.device, dtype=torch.int32)
                self._scan_scratch[key] = t
            return t

        # Build level sizes: n0=n, n1=ceil(n0/tg), ... until <= tg.
        sizes: list[int] = [int(n)]
        while sizes[-1] > tg:
            sizes.append((sizes[-1] + tg - 1) // tg)

        # Per-level outputs and block sums.
        # out_levels[i] is exclusive scan of input at level i (length sizes[i]).
        out_levels: list["Tensor"] = [starts_u32_i32[:n]]  # view: writes into `starts_u32_i32`
        block_sums_levels: list["Tensor"] = []

        for li, sz in enumerate(sizes[1:], start=1):
            out_levels.append(_scratch(f"scan_out_L{li}", sz))
        for li, sz in enumerate(sizes[:-1]):
            nb = (sz + tg - 1) // tg
            block_sums_levels.append(_scratch(f"scan_block_sums_L{li}", nb))

        # Forward: scan each level and produce block sums for next level.
        in_level = counts_u32_i32
        for li, sz in enumerate(sizes):
            out_i = out_levels[li]
            if li == len(sizes) - 1:
                tmp1 = _scratch("scan_tmp1", 1)
                self.ops.exclusive_scan_u32_pass1(in_level, out_i, tmp1, sz)
            else:
                bs = block_sums_levels[li]
                self.ops.exclusive_scan_u32_pass1(in_level, out_i, bs, sz)
                in_level = bs

        # Backward: add scanned block offsets to each lower level.
        for li in range(len(sizes) - 2, -1, -1):
            self.ops.exclusive_scan_u32_add_block_offsets(out_levels[li], out_levels[li + 1], sizes[li])

        # Finalize total sum into starts[n].
        self.ops.exclusive_scan_u32_finalize_total(counts_u32_i32, starts_u32_i32, n)
    
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
        # Derive a stable step size from current state (pseudo-time).
        dt = self._derive_dt(velocities, masses)
        self.last_dt = float(dt)

        # Optional finite checks (debug; correctness-first).
        if not hasattr(self, "_debug_step"):
            self._debug_step = 0
        self._debug_step += 1
        do_dbg = bool(getattr(self.config, "debug_check_finite", False)) and (
            (self._debug_step % max(1, int(getattr(self.config, "debug_check_finite_every", 1)))) == 0
        )

        def _assert_finite(tag: str, t: "Tensor") -> None:
            if not do_dbg:
                return
            if t.numel() == 0:
                return
            finite = torch.isfinite(t)
            ok = bool(finite.all().detach().item())
            if ok:
                return
            bad = int((~finite).sum().detach().item())
            # Include a couple of scalar summaries to help narrow overflow vs NaN sources.
            t32 = t.detach().to(torch.float32)
            with torch.no_grad():
                max_abs = float(t32.abs().max().detach().item())
                min_v = float(t32.min().detach().item())
                max_v = float(t32.max().detach().item())
            raise RuntimeError(
                f"[ManifoldPhysics] non-finite detected at stage='{tag}' "
                f"(debug_step={self._debug_step}, bad={bad}/{t.numel()}, "
                f"min={min_v:.6g}, max={max_v:.6g}, max_abs={max_abs:.6g})"
            )

        _assert_finite("input.positions", positions)
        _assert_finite("input.velocities", velocities)
        _assert_finite("input.energies", energies)
        _assert_finite("input.heats", heats)
        _assert_finite("input.masses", masses)

        # Optional accounting snapshot (CPU scalars via `.item()` → device sync).
        # Keep disabled by default for performance.
        if bool(getattr(self.config, "enable_energy_report", False)):
            if energies.numel() > 0:
                v2 = velocities.to(torch.float32).pow(2).sum(dim=1)
                ke = 0.5 * masses.to(torch.float32) * v2
                e_int = energies.to(torch.float32)
                q = heats.to(torch.float32)
                self.last_energy_report = {
                    "ke_sum": float(ke.sum().detach().item()),
                    "e_sum": float(e_int.sum().detach().item()),
                    "heat_sum": float(q.sum().detach().item()),
                }
            else:
                self.last_energy_report = {"ke_sum": 0.0, "e_sum": 0.0, "heat_sum": 0.0}
        else:
            self.last_energy_report = None

        # 1. Scatter particles to fields
        self.scatter_particles(positions, masses, heats, energies)
        _assert_finite("after.scatter.gravity_field", self.gravity_field)
        _assert_finite("after.scatter.heat_field", self.heat_field)
        
        # 2. Solve field equations
        self.solve_gravity()
        _assert_finite("after.solve_gravity.gravity_potential", self.gravity_potential)

        # Heat diffusion (spectral / FFT, periodic domain):
        # Exact per-step update in frequency space:
        #   T̂ <- exp(-α k² dt) * T̂
        cfg = self.config
        self._ensure_fft_poisson_cache()
        assert self._fft_k2 is not None

        # [CHOICE] thermal diffusion coefficient (mean-field, GPU-only)
        # [FORMULA] α = κ / (ρ̄ c_v)
        # [REASON] thermal diffusivity is derived from conductivity κ and heat capacity
        # [NOTES] Uses mean density ρ̄ to keep FFT update exact/fast. Variable-coefficient
        #         diffusion (α(x)) requires a real-space stencil (TODO).
        if self.temperature_field.numel() > 0:
            gx, gy, gz = self.grid_dims
            h = float(cfg.grid_spacing)
            vol = float(gx * gy * gz) * (h * h * h)
            total_mass = self.gravity_field.to(torch.float32).sum()  # device scalar

            # [CHOICE] vacuum handling for diffusion
            # [FORMULA] if ρ̄ == 0 then α = 0 (no medium to conduct heat)
            # [REASON] defines vacuum behavior explicitly (no eps clamps)
            rho_bar = total_mass / vol
            denom = rho_bar * float(cfg.specific_heat)  # device scalar
            kappa = torch.tensor(float(cfg.thermal_conductivity), device=self.device, dtype=torch.float32)
            alpha = torch.where(denom > 0, kappa / denom, torch.zeros_like(denom))

            T = self.temperature_field.to(torch.float32)
            T_hat = torch.fft.fftn(T)
            decay = torch.exp((-(alpha * float(dt))) * self._fft_k2)
            T_next = torch.fft.ifftn(T_hat * decay).real.to(self.dtype)
            self.temperature_field.copy_(T_next)
            # [CHOICE] enforce non-negative temperature (physical boundary)
            # [FORMULA] T_cell := max(T_cell, 0)
            # [REASON] absolute temperature cannot be negative; fp32 spectral diffusion
            #          can produce tiny negative roundoff artifacts that would otherwise
            #          cascade into invalid sqrt(T) in the drag model.
            # [NOTES] this is a boundary condition projection, not a tunable clamp.
            self.temperature_field.clamp_min_(0.0)
        _assert_finite("after.diffuse.temperature_field", self.temperature_field)
        
        # 3. Gather from fields and update particle state
        # Gather/update uses the derived dt.
        positions, velocities, energies, heats, excitations = self.gather_update_particles(
            positions, velocities, energies, heats, excitations, masses, dt=float(dt)
        )
        _assert_finite("after.gather.positions", positions)
        _assert_finite("after.gather.velocities", velocities)
        _assert_finite("after.gather.energies", energies)
        _assert_finite("after.gather.heats", heats)
        
        # 4. Particle-particle interactions (collision + excitation transfer)
        # Auto-select collision algorithm based on particle count
        n = positions.size(0)
        if use_spatial_hash is None:
            use_spatial_hash = n >= 1000
        
        if use_spatial_hash:
            # O(N) spatial hash - better for large particle counts
            velocities, excitations, heats = self.compute_interactions_spatial_hash(
                positions, velocities, excitations, masses, heats, dt=float(dt)
            )
        else:
            # O(N²) brute force - lower overhead for small counts
            velocities, excitations, heats = self.compute_interactions(
                positions, velocities, excitations, masses, heats, dt=float(dt)
            )

        _assert_finite("after.collisions.velocities", velocities)
        _assert_finite("after.collisions.heats", heats)
        
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
        self.config.validate_invariants()
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
        # Snapshot of the carrier count used for stable gating within a tick.
        # This avoids host `.item()` synchronization to discover the current count.
        self._num_carriers_snapshot = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.num_carriers = 0

        # Random phases used when spawning carriers inside the kernel
        self._random_phases = torch.rand(self.max_carriers, device=self.device, dtype=self.dtype)

        # Global energy statistics buffer (GPU-written each step):
        # [mean_abs, mean, std, count]
        self._energy_stats = torch.zeros(4, device=self.device, dtype=self.dtype)

        # Carrier accumulators (force_r, force_i, w_sum, w_omega, w_amp, off_score, off_idx)
        # 8 fields per carrier (added w_omega2_sum for adaptive gate width).
        self._carrier_accums = torch.zeros(self.max_carriers * 8, device=self.device, dtype=torch.int32)

        # ---------------------------------------------------------------------
        # Carrier frequency binning (GPU-only)
        # ---------------------------------------------------------------------
        # [CHOICE] omega-bin indexing for sparse coupling
        # [FORMULA] bin = floor((ω - ω_min) / W), with W derived on-device
        # [REASON] reduces dense O(N·M) coupling loops by only visiting carriers
        #          whose tuning weight can be nonzero in fp32.
        # [NOTES] - No host sync: ω_min/W computed via GPU reductions.
        #         - num_bins is fixed to max_carriers (capacity-derived, not tunable).
        self._omega_min_key = torch.full((1,), -1, device=self.device, dtype=torch.int32)  # 0xFFFFFFFF
        self._omega_max_key = torch.zeros((1,), device=self.device, dtype=torch.int32)
        # bin_params = [omega_min, inv_bin_width] (fp32) written on GPU
        self._bin_params = torch.zeros((2,), device=self.device, dtype=self.dtype)
        self._bin_counts = torch.zeros((self.max_carriers,), device=self.device, dtype=torch.int32)
        self._bin_starts = torch.zeros((self.max_carriers + 1,), device=self.device, dtype=torch.int32)
        self._bin_offsets = torch.zeros((self.max_carriers,), device=self.device, dtype=torch.int32)
        self._carrier_binned_idx = torch.zeros((self.max_carriers,), device=self.device, dtype=torch.int32)
        self._bin_scan_scratch: dict[tuple[str, int], "Tensor"] = {}

        # RNG seed for Langevin noise (host-controlled)
        self._rng_seed: int = 1

        # Load Metal ops
        from .jit import load_caramba_metal_ops
        self._ops = load_caramba_metal_ops()

    def synchronize(self) -> None:
        """Force completion of queued GPU work (execution barrier)."""
        if self.device.type == "mps":
            torch.mps.synchronize()

    def done_thinking(self) -> bool:
        """Return True when the spectral system is sufficiently settled.

        [CHOICE] settledness test (carrier coherence)
        [FORMULA] done := (∃ crystallized carrier) OR (min(conflict) <= θ_crys)
        [REASON] A settled spectral field has at least one coherent, persistent mode.
        [NOTES] Uses `.item()` reductions (device sync). Call sparingly.
        """
        # Ensure snapshot reflects latest kernel updates without host `.item()` in the hot path.
        self._num_carriers_snapshot.copy_(self._num_carriers_buf)
        n = int(self._num_carriers_snapshot.detach().item())
        if n <= 0:
            return False

        cfg = self.config
        state = self.carrier_state[:n]
        if bool((state == 2).any().detach().item()):
            return True

        conflict = self.carrier_conflict[:n]
        min_conf = float(conflict.min().detach().item())
        return min_conf <= float(cfg.crystallize_conflict_threshold)

    @property
    def ops(self):
        return self._ops

    def _exclusive_scan_u32_into_starts(
        self, counts_u32_i32: "Tensor", starts_u32_i32: "Tensor", n: int
    ) -> None:
        """Exclusive scan of int32 tensor (uint32 semantics) into starts[:n] and starts[n]=total."""
        if n <= 0:
            starts_u32_i32.zero_()
            return

        tg = 256

        def _scratch(name: str, size: int) -> "Tensor":
            key = (name, int(size))
            t = self._bin_scan_scratch.get(key)
            if t is None or t.numel() != size or t.device != counts_u32_i32.device:
                t = torch.empty((size,), device=counts_u32_i32.device, dtype=torch.int32)
                self._bin_scan_scratch[key] = t
            return t

        sizes: list[int] = [int(n)]
        while sizes[-1] > tg:
            sizes.append((sizes[-1] + tg - 1) // tg)

        out_levels: list["Tensor"] = [starts_u32_i32[:n]]
        block_sums_levels: list["Tensor"] = []
        for li, sz in enumerate(sizes[1:], start=1):
            out_levels.append(_scratch(f"scan_out_L{li}", sz))
        for li, sz in enumerate(sizes[:-1]):
            nb = (sz + tg - 1) // tg
            block_sums_levels.append(_scratch(f"scan_block_sums_L{li}", nb))

        in_level = counts_u32_i32
        for li, sz in enumerate(sizes):
            out_i = out_levels[li]
            if li == len(sizes) - 1:
                tmp1 = _scratch("scan_tmp1", 1)
                self.ops.exclusive_scan_u32_pass1(in_level, out_i, tmp1, sz)
            else:
                bs = block_sums_levels[li]
                self.ops.exclusive_scan_u32_pass1(in_level, out_i, bs, sz)
                in_level = bs
        for li in range(len(sizes) - 2, -1, -1):
            self.ops.exclusive_scan_u32_add_block_offsets(out_levels[li], out_levels[li + 1], sizes[li])
        self.ops.exclusive_scan_u32_finalize_total(counts_u32_i32, starts_u32_i32, n)

    def _build_carrier_bins(self) -> None:
        """Build carrier ω-bins into `_carrier_binned_idx` and `_bin_starts` (GPU-only)."""
        num_bins = int(self.max_carriers)
        # Reset min/max keys for the reduction.
        self._omega_min_key.fill_(-1)
        self._omega_max_key.zero_()

        self.ops.spectral_reduce_omega_minmax_keys(
            self.carrier_omega,
            self._num_carriers_snapshot,
            self._omega_min_key,
            self._omega_max_key,
        )
        self.ops.spectral_compute_bin_params(
            self._omega_min_key,
            self._omega_max_key,
            self._num_carriers_snapshot,
            self._bin_params,
            float(self.config.gate_width_max),
        )

        # Count carriers per bin, scan to starts, and scatter carrier indices.
        self._bin_counts.zero_()
        self.ops.spectral_bin_count_carriers(
            self.carrier_omega,
            self._num_carriers_snapshot,
            self._bin_counts,
            self._bin_params,
            num_bins,
        )
        self._exclusive_scan_u32_into_starts(self._bin_counts, self._bin_starts, num_bins)
        self._bin_offsets.copy_(self._bin_starts[:num_bins])
        self.ops.spectral_bin_scatter_carriers(
            self.carrier_omega,
            self._num_carriers_snapshot,
            self._bin_offsets,
            self._bin_params,
            num_bins,
            self._carrier_binned_idx,
        )

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
            anchor_eps = float(self.config.anchor_random_eps) * 0.25
            rand_energy_eps = float(self.config.topdown_random_energy_eps) * 0.25
            offender_floor = float(self.config.offender_weight_floor)
            repulsion = 0.0
        elif mode_s in ("disambiguate", "resolve", "separate"):
            m = 2
            anchor_eps = float(self.config.anchor_random_eps) * 0.50
            rand_energy_eps = float(self.config.topdown_random_energy_eps) * 0.50
            offender_floor = float(self.config.offender_weight_floor)
            repulsion = float(self.config.repulsion_scale)
        elif mode_s in ("explore", "counterfactual", "counterfactual_exploration"):
            m = 3
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

            # Snapshot current carrier count (GPU→GPU copy, no host sync).
            self._num_carriers_snapshot.copy_(self._num_carriers_buf)
            self._build_carrier_bins()
            self._random_phases.uniform_()
            self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
            cfg = self.config

            self._carrier_accums.zero_()
            self.ops.spectral_accumulate_forces(
                osc_phase,
                osc_omega,
                osc_amp,
                self.carrier_omega,
                self.carrier_gate_width,
                self.carrier_conflict,
                self._carrier_accums,
                self._bin_starts,
                self._carrier_binned_idx,
                self._bin_params,
                int(self.max_carriers),
                int(osc_phase.shape[0]),
                self._num_carriers_snapshot,
                int(self.max_carriers),
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
                float(offender_floor),
                float(cfg.conflict_threshold),
            )

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
                int(self.max_carriers),
                self._num_carriers_snapshot,
                float(self.dt),
                float(cfg.coupling_scale),
                float(cfg.carrier_reg),
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
                self._carrier_accums,
            )

            self.ops.spectral_topdown_bias_energies(
                energy,
                osc_amp,
                self.carrier_state,
                self.carrier_anchor_idx,
                self.carrier_anchor_weight,
                self._num_carriers_snapshot,
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
                self._num_carriers_snapshot,
                self._bin_starts,
                self._carrier_binned_idx,
                self._bin_params,
                int(self.max_carriers),
                int(self.max_carriers),
                float(self.dt),
                float(cfg.coupling_scale),
                int(self._rng_seed) & 0xFFFFFFFF,
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
                float(cfg.crystallized_coupling_boost),
                float(cfg.topdown_phase_scale),
            )

            self.ops.spectral_spawn_uncoupled(
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
                int(self.max_carriers),
                self._num_carriers_snapshot,
                float(cfg.uncoupled_threshold),
                float(cfg.gate_width_init),
                float(cfg.gate_width_min),
                float(cfg.gate_width_max),
                self._bin_starts,
                self._carrier_binned_idx,
                self._bin_params,
                int(self.max_carriers),
            )
            # Report full-capacity carrier buffers + a device-side count snapshot.
            cr = self.carrier_real
            ci = self.carrier_imag
            amp = torch.sqrt(cr * cr + ci * ci)
            phase = torch.atan2(ci, cr)
            out = {
                "num_carriers": self._num_carriers_snapshot.clone(),
                "frequencies": self.carrier_omega,
                "gate_widths": self.carrier_gate_width,
                "amplitudes": amp,
                "phases": phase,
                "conflict": self.carrier_conflict,
                "osc_phase": osc_phase,
                "osc_energy": energy,
                "carrier_state": self.carrier_state,
                "carrier_age": self.carrier_age,
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

        # Snapshot current carrier count (GPU→GPU copy; avoids host `.item()` sync).
        self._num_carriers_snapshot.copy_(self._num_carriers_buf)
        self._build_carrier_bins()
        self._random_phases.uniform_()

        cfg = self.config
        # Advance RNG seed so noise changes each step deterministically
        self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF

        # 1. Clear accumulators
        self._carrier_accums.zero_()

        # 2. Accumulate forces (parallel over oscillators)
        self.ops.spectral_accumulate_forces(
            osc_phase,
            osc_omega,
            osc_amp,
            self.carrier_omega,
            self.carrier_gate_width,
            self.carrier_conflict,
            self._carrier_accums,
            self._bin_starts,
            self._carrier_binned_idx,
            self._bin_params,
            int(self.max_carriers),
            int(osc_phase.shape[0]),
            self._num_carriers_snapshot,
            int(self.max_carriers),
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
            float(cfg.offender_weight_floor),
            float(cfg.conflict_threshold)
        )

        # 3. Update carriers (parallel over carriers, using accumulated forces)
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
            int(self.max_carriers),
            self._num_carriers_snapshot,
            float(self.dt),
            float(cfg.coupling_scale),
            float(cfg.carrier_reg),
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
            self._carrier_accums, # Pass accumulators
        )

        # Top-down energy bias (crystallized carriers act as priors/completions).
        # NOTE: This updates `energy` (a local fp32 view); callers can choose to
        # use the returned "osc_energy" for downstream inference.
        self.ops.spectral_topdown_bias_energies(
            energy,
            osc_amp,
            self.carrier_state,
            self.carrier_anchor_idx,
            self.carrier_anchor_weight,
            self._num_carriers_snapshot,
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
            self._num_carriers_snapshot,
            self._bin_starts,
            self._carrier_binned_idx,
            self._bin_params,
            int(self.max_carriers),
            int(self.max_carriers),
            float(self.dt),
            float(cfg.coupling_scale),
            int(self._rng_seed) & 0xFFFFFFFF,
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
            float(cfg.crystallized_coupling_boost),
            float(cfg.topdown_phase_scale),
        )

        # Spawn carriers for any uncoupled oscillators
        # (ensures every oscillator is coupled to at least one carrier)
        self.ops.spectral_spawn_uncoupled(
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
            int(self.max_carriers),
            self._num_carriers_snapshot,
            float(cfg.uncoupled_threshold),  # oscillators with total coupling < this spawn
            float(cfg.gate_width_init),
            float(cfg.gate_width_min),
            float(cfg.gate_width_max),
            self._bin_starts,
            self._carrier_binned_idx,
            self._bin_params,
            int(self.max_carriers),
        )

        # Prepare full-capacity state views for dashboard + device-side count snapshot.
        cr = self.carrier_real
        ci = self.carrier_imag
        amp = torch.sqrt(cr * cr + ci * ci)
        phase = torch.atan2(ci, cr)

        return {
            "num_carriers": self._num_carriers_snapshot.clone(),
            "frequencies": self.carrier_omega,
            "gate_widths": self.carrier_gate_width,
            "amplitudes": amp,
            "phases": phase,
            "conflict": self.carrier_conflict,
            "osc_phase": osc_phase,
            "osc_energy": energy,
            "carrier_state": self.carrier_state,
            "carrier_age": self.carrier_age,
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
