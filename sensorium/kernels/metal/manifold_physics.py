"""Metal manifold physics kernels.

Single implementation using Metal acceleration. No fallbacks.
If Metal is not available or something fails, we raise an exception.

------------------------------------
COMMENT CONVENTION (physics choices)
------------------------------------
For clarity, and reviewer ergonomics, all values, methods, and equations
must be annotated to clarify the choices made, the formula used, and the
reasoning behind the choice. Always think about reviewer ergonomics, and
provide additional context to the person who has to read through this dense
code, so they are informed enough to make high-quality decisions where they
agree, or disagree, which is valuable feedback to improve the system.

Format:
  # [CHOICE] <name>
  # [FORMULA] <math / equation / mapping>
  # [REASON] <brief why this form/value>
  # [NOTES] <brief caveats, assumptions, invariants, TODOs>

All effort should be made to achieve:
  - maximum correctness: make the modeled physics explicit and falsifiable
  - maximum performance: make optimizations explicit and non-semantic
In that order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from sensorium.kernels.physics_units import (
    PhysicalConstants,
    UnitSystem,
    assert_finite_constants,
    gas_R_specific_sim,
    dynamic_viscosity_sim,
)

from sensorium.kernels.pic import (
    cic_stencil_periodic,
    gather_trilinear,
    gather_trilinear_vec3,
)

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class GasNumerics:
    """Numerics policy for the gas grid update (dual-energy internal energy).

    The torch reference implementation lives in `sensorium/kernels/gas_dynamics.py`,
    but the Metal domain should not depend on that module.
    """

    cfl: float = 0.4
    cfl_diffusion: float = 0.15
    rho_min: float = 1e-3
    p_min: float = 1e-3


@dataclass(frozen=True)
class ThermodynamicsDomainConfig:
    """Public config surface for tests / backend symmetry.

    The Metal backend intentionally keeps most physics constants fixed; the
    only tunables are the grid topology and a soft timestep ceiling.
    """

    grid_size: tuple[int, int, int] = (64, 64, 64)
    dt_max: float = 0.015


@dataclass(frozen=True)
class CoherenceFieldConfig:
    """Compatibility config surface for coherence field tests.

    The production Metal `OmegaWaveDomain` intentionally derives its ω-lattice
    size and parameters. Tests (and backend symmetry) use an explicit config.
    """

    omega_bins: int = 64
    omega_min: float = -4.0
    omega_max: float = 4.0
    hbar_eff: float = 1.0
    mass_eff: float = 1.0
    g_interaction: float = -1.0
    energy_decay: float = 0.0


def thermodynamics_domain_available() -> bool:
    """Check if the Metal thermodynamics domain is available."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal backend not available")
    try:
        from .jit import load_manifold_metal_ops

        load_manifold_metal_ops()
        return True
    except Exception as err:
        raise RuntimeError("Metal thermodynamics domain is not available") from err


def _derived_grid_spacing(grid_size: tuple[int, int, int]) -> float:
    """Derive Δx from grid topology (no exposed knobs).

    [CHOICE] domain length is 1.0 (simulation length units)
    [FORMULA] Δx = 1 / max(gx,gy,gz)
    [REASON] keeps the spatial domain normalized; changing `grid_size` is the
             only way to change spatial resolution / physical scale.
    """
    g = int(max(grid_size))
    if g <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    return 1.0 / float(g)


def _morton_part1by2_u32(v: "Tensor") -> "Tensor":
    """Bit-interleave helper for 3D Morton code (Z-order curve).

    [CHOICE] 10-bit per axis Morton (supports up to 1024^3 grids)
    [FORMULA] morton = part(x) | (part(y)<<1) | (part(z)<<2)
    [REASON] our grids are <= O(10^2)^3 today; 10-bit is ample and keeps the key
             compact and deterministic, with no hashing/collisions.
    [NOTES] `v` is masked to 10 bits defensively.
    """
    v = v.to(torch.int64) & 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _morton3_u32(ix: "Tensor", iy: "Tensor", iz: "Tensor") -> "Tensor":
    """Vectorized 3D Morton code for integer grid coordinates."""
    return (
        _morton_part1by2_u32(ix)
        | (_morton_part1by2_u32(iy) << 1)
        | (_morton_part1by2_u32(iz) << 2)
    )


# -------------------------------------------------------------------------
# Material model (single choice, not a config surface)
# -------------------------------------------------------------------------
# These are material properties (medium choice), not tunable parameters.
# If/when we support multiple media, this becomes a discrete selection, not a knob.
_AIR_MOLAR_MASS_KG_PER_MOL: float = 0.02897  # dry air
_AIR_GAMMA: float = 1.4
_AIR_DYNAMIC_VISCOSITY: float = 1.8e-5  # Pa·s in SI
_AIR_PRANDTL: float = 0.71

_PIC_SCATTER_BACKEND: str = os.getenv("SENSORIUM_PIC_SCATTER", "metal").strip().lower()


def _read_metal_mode_anchors() -> int:
    """Read MODE_ANCHORS value from the Metal source file.

    This creates a mechanical link between Python and Metal configs,
    preventing silent drift if someone changes one but not the other.
    """
    import re
    from pathlib import Path

    metal_file = Path(__file__).parent / "manifold_physics.metal"
    if not metal_file.exists():
        raise RuntimeError(f"Metal source not found at {metal_file}")
    content = metal_file.read_text()
    match = re.search(r"#define\s+MODE_ANCHORS\s+(\d+)u?", content)
    if not match:
        raise RuntimeError("Could not find MODE_ANCHORS define in Metal source")
    return int(match.group(1))


# Cache the Metal value at module load time (fail fast).
_METAL_MODE_ANCHORS: int = _read_metal_mode_anchors()


class ThermodynamicsDomain:
    """Metal-accelerated thermodynamics domain (compressible ideal-gas NS + PIC).

    No fallbacks. Metal only. Exceptions on failure.
    """

    def __init__(
        self,
        config: Optional[ThermodynamicsDomainConfig] = None,
        grid_size: tuple[int, int, int] = (64, 64, 64),
        *,
        dt_max: Optional[float] = None,
        device: str = "mps",
    ):
        # Backward-compat: allow `ThermodynamicsDomain((gx,gy,gz))`.
        if config is not None and isinstance(config, tuple):  # type: ignore[unreachable]
            grid_size = config  # type: ignore[assignment]
            config = None

        if device != "mps":
            raise RuntimeError(
                f"Metal ThermodynamicsDomain requires device='mps' (got {device!r})"
            )

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")

        self.device = torch.device("mps")
        self.dtype = torch.float32

        if config is not None:
            grid_size = config.grid_size
            dt_max_val = float(config.dt_max)
        else:
            dt_max_val = float(dt_max) if dt_max is not None else 0.015

        gx, gy, gz = grid_size
        self.grid_dims = (int(gx), int(gy), int(gz))

        self.dt_max: float = float(dt_max_val)
        self.config = ThermodynamicsDomainConfig(
            grid_size=self.grid_dims, dt_max=self.dt_max
        )

        # ---------------------------------------------------------------------
        # Discretization + constants (no config surface)
        # ---------------------------------------------------------------------
        self.grid_spacing: float = _derived_grid_spacing(self.grid_dims)

        # [CHOICE] unit system (simulation units → SI)
        # [REASON] ω is defined in natural inverse-time units with ω∈[0,2), and the
        #          Planck exchange uses ħ=k_B=1. We therefore choose a deterministic
        #          natural unit system where (ħ_sim=k_B_sim=1) and c_v=1 for the
        #          chosen γ and medium. This makes the thermal crossover ω≈T occur
        #          at T∼O(1), as intended.
        # [CHOICE] material model (single medium: dry air, ideal gas)
        # [REASON] these are material properties, not tunable parameters
        self.molecular_weight_kg_per_mol: float = float(_AIR_MOLAR_MASS_KG_PER_MOL)
        self.gamma: float = float(_AIR_GAMMA)
        # Store SI property and its derived simulation-unit value separately.
        self.dynamic_viscosity_si: float = float(_AIR_DYNAMIC_VISCOSITY)  # Pa·s
        self.prandtl: float = float(_AIR_PRANDTL)

        # Derive the natural unit system after the medium choice is fixed.
        self.unit_system: UnitSystem = UnitSystem.omega_natural(
            gamma=float(self.gamma),
            molecular_weight_kg_per_mol=float(self.molecular_weight_kg_per_mol),
            length_unit_m=1.0,
        )
        self.constants: PhysicalConstants = PhysicalConstants.from_codata_si(
            self.unit_system
        )
        assert_finite_constants(self.constants)

        # ---------------------------------------------------------------------
        # Transport coefficients (simulation-scale, not SI-scale)
        # ---------------------------------------------------------------------
        # NOTE:
        # We run in a normalized ω-natural unit system (ħ=k_B=1, ω∈O(1), domain length=1).
        # Converting SI μ (Pa·s) into these units yields astronomically large values because
        # the derived base units imply a *tiny* mass unit and a *huge* time unit; the result
        # makes explicit diffusion terms numerically explosive at our grid spacing.
        #
        # Therefore, for this normalized simulation we treat viscosity/conductivity as a
        # grid-scale submodel coefficient (still a model choice, but *not* a per-run knob).
        # If/when we simulate a specific real-world scale, we can revisit SI transport.
        self.dynamic_viscosity: float = 1e-4

        # Optional gravitational potential Φ (periodic Poisson solve).
        self.gravity_potential = torch.zeros(
            gx, gy, gz, device=self.device, dtype=self.dtype
        )

        # ---------------------------------------------------------------------
        # Compressible gas (ideal-gas Navier–Stokes): conserved grid state
        # ---------------------------------------------------------------------
        # Conserved / carried per-volume quantities:
        #   rho:    mass density
        #   mom:    momentum density (rho * u)
        #   e_int:  internal energy density (rho e)
        #
        # [CHOICE] store internal energy density ("Path B" / dual-energy)
        # [FORMULA] e_int := rho * e   (no kinetic energy term)
        # [REASON] avoids catastrophic cancellation in high-Mach / vacuum-adjacent regimes
        #          where total energy and kinetic energy are large and nearly equal.
        # [NOTES] Any oscillator↔thermal exchange must be explicit and locally
        #         energy-conserving (not implicit via repeated re-scatter).
        self.rho_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.mom_field = torch.zeros(
            gx, gy, gz, 3, device=self.device, dtype=self.dtype
        )
        self.e_int_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        # NOTE: Metal ABI uses the name `E_field` for internal energy density (rho e).
        self.E_field = self.e_int_field

        # Numerics policy for positivity/CFL (used by the gas solver).
        self.gas_numerics = GasNumerics()

        # Diagnostics (optional, lightweight)
        self.last_dt: float = float(self.grid_spacing)
        self.last_energy_report: Optional[dict[str, Any]] = None

        # FFT Poisson cache (periodic domain).
        self._fft_k2: Optional[torch.Tensor] = None  # (gx, gy, gz) float32 on MPS
        self._fft_inv_k2: Optional[torch.Tensor] = None  # (gx, gy, gz) float32 on MPS

        # Collision snapshots (avoid write hazards by reading old state).
        self._vel_in: Optional[torch.Tensor] = None  # (N, 3)
        self._heat_in: Optional[torch.Tensor] = None  # (N,)

        # Spatial-hash scratch buffers (reused to avoid per-step allocations).
        # Shapes depend on N and on derived hash grid dims (num_cells).
        self._hash_particle_cell_idx: Optional[torch.Tensor] = None  # (N,) int32
        self._hash_sorted_particle_idx: Optional[torch.Tensor] = None  # (N,) int32
        self._hash_cell_counts: Optional[torch.Tensor] = (
            None  # (num_cells,) int32 (must be zeroed each call)
        )
        self._hash_cell_starts: Optional[torch.Tensor] = None  # (num_cells+1,) int32
        self._hash_cell_offsets: Optional[torch.Tensor] = (
            None  # (num_cells,) int32 (working copy of starts)
        )

        # Sort-based scatter scratch buffers (for "sorted" mode).
        self._sort_particle_cell_idx: Optional[torch.Tensor] = None  # (N,) uint32
        self._sort_cell_counts: Optional[torch.Tensor] = None  # (num_cells,) uint32
        self._sort_cell_starts: Optional[torch.Tensor] = None  # (num_cells,) uint32
        self._sort_cell_offsets: Optional[torch.Tensor] = None  # (num_cells,) uint32
        self._sort_pos_out: Optional[torch.Tensor] = None  # (N, 3) float32
        self._sort_vel_out: Optional[torch.Tensor] = None  # (N, 3) float32
        self._sort_mass_out: Optional[torch.Tensor] = None  # (N,) float32
        self._sort_heat_out: Optional[torch.Tensor] = None  # (N,) float32
        self._sort_energy_out: Optional[torch.Tensor] = None  # (N,) float32
        self._sort_original_idx: Optional[torch.Tensor] = None  # (N,) uint32

        # Metal kernel bindings (JIT compiled, cached per-process).
        from .jit import load_manifold_metal_ops

        self._ops = load_manifold_metal_ops()

        # ---------------------------------------------------------------------
        # Kernel "log book" (GPU debug events)
        # ---------------------------------------------------------------------
        # Default ON (can disable with SENSORIUM_KERNEL_LOG=0).
        env_on = os.getenv("SENSORIUM_KERNEL_LOG", "1").strip().lower()
        self.kernel_log_enabled: bool = env_on not in ("0", "false", "no", "off")
        self.kernel_log_capacity: int = int(
            os.getenv("SENSORIUM_KERNEL_LOG_CAP", "2048")
        )
        if self.kernel_log_capacity < 0:
            self.kernel_log_capacity = 0
        cap_alloc = max(1, self.kernel_log_capacity if self.kernel_log_enabled else 1)
        # `dbg_head` is a single u32 counter (stored as int32 tensor).
        self._dbg_head = torch.zeros((1,), device=self.device, dtype=torch.int32)
        # `dbg_words` stores cap * 6 u32 words (stored as int32 tensor).
        self._dbg_words = torch.zeros(
            (cap_alloc * 6,), device=self.device, dtype=torch.int32
        )

        # Gas grid RK2 scratch buffers (allocated lazily).
        self._gas_rho1: Optional[torch.Tensor] = None
        self._gas_mom1: Optional[torch.Tensor] = None
        self._gas_e1: Optional[torch.Tensor] = None
        self._gas_rho2: Optional[torch.Tensor] = None
        self._gas_mom2: Optional[torch.Tensor] = None
        self._gas_e2: Optional[torch.Tensor] = None
        self._gas_k1_rho: Optional[torch.Tensor] = None
        self._gas_k1_mom: Optional[torch.Tensor] = None
        self._gas_k1_e: Optional[torch.Tensor] = None

    def _decode_kernel_log(self, tail: int = 24) -> str:
        """Decode the GPU log book into a short human-readable string."""
        if not (self.kernel_log_enabled and self.kernel_log_capacity > 0):
            return ""
        try:
            n_ev = int(self._dbg_head.detach().to("cpu").item())
            n_ev = max(0, min(n_ev, int(self.kernel_log_capacity)))
            if n_ev <= 0:
                return ""
            words_i32 = (
                self._dbg_words.detach().to("cpu").numpy().astype(np.uint32, copy=False)
            )
            words = words_i32[: n_ev * 6].reshape(n_ev, 6)
            payload_u32 = words[:, 2:6].reshape(-1)
            payload = payload_u32.view(np.float32).reshape(n_ev, 4)
            names = {
                0x01: "PIC base",
                0x02: "PIC vacuum",
                0x03: "PIC bad cv",
                0x04: "PIC bad e_int",
                0x05: "PIC bad T/heat",
                0x06: "PIC bad advect",
                0x07: "PIC bad consv",
                0x12: "GAS bad U1",
                0x13: "GAS bad U2",
                0x20: "GAS bad RHS1",
                0x21: "GAS bad RHS2",
            }
            start = max(0, n_ev - int(tail))
            lines = []
            for i in range(start, n_ev):
                tag = int(words[i, 0])
                gid = int(words[i, 1])
                a, b, c, d = (
                    float(payload[i, 0]),
                    float(payload[i, 1]),
                    float(payload[i, 2]),
                    float(payload[i, 3]),
                )
                nm = names.get(tag, f"tag=0x{tag:02x}")
                lines.append(
                    f"{i:04d} {nm:>12} gid={gid:<6} a={a:.6g} b={b:.6g} c={c:.6g} d={d:.6g}"
                )
            return "\n".join(lines)
        except Exception:
            return "(kernel log decode failed)"

    def _R_specific(self) -> float:
        """Gas constant for the chosen material in simulation units."""
        return float(
            gas_R_specific_sim(
                self.unit_system,
                molecular_weight_kg_per_mol=float(self.molecular_weight_kg_per_mol),
            )
        )

    def _c_v(self) -> float:
        """Specific heat at constant volume from (γ, R): c_v = R/(γ-1)."""
        return float(self._R_specific()) / (float(self.gamma) - 1.0)

    def _c_p(self) -> float:
        """Specific heat at constant pressure: c_p = γ R/(γ-1)."""
        g = float(self.gamma)
        return (g * float(self._R_specific())) / (g - 1.0)

    def _thermal_conductivity(self) -> float:
        """Thermal conductivity κ derived from (μ, c_p, Pr): κ = μ c_p / Pr.

        [CHOICE] κ from Prandtl relation for ideal gas
        [FORMULA] κ = μ c_p / Pr
        [REASON] eliminates a separate knob; uses existing material properties
        """
        mu = float(self.dynamic_viscosity)
        Pr = float(self.prandtl)
        if not (mu > 0.0) or not (Pr > 0.0):
            return 0.0
        return (mu * float(self._c_p())) / Pr

    def _ensure_gas_rk2_scratch(self) -> None:
        """Allocate/reuse RK2 scratch buffers for the gas grid step."""
        dev = self.device
        dtype = self.dtype
        if (
            self._gas_rho1 is None
            or self._gas_rho1.shape != self.rho_field.shape
            or self._gas_rho1.device != dev
        ):
            self._gas_rho1 = torch.empty_like(self.rho_field, device=dev, dtype=dtype)
        if (
            self._gas_mom1 is None
            or self._gas_mom1.shape != self.mom_field.shape
            or self._gas_mom1.device != dev
        ):
            self._gas_mom1 = torch.empty_like(self.mom_field, device=dev, dtype=dtype)
        if (
            self._gas_e1 is None
            or self._gas_e1.shape != self.e_int_field.shape
            or self._gas_e1.device != dev
        ):
            self._gas_e1 = torch.empty_like(self.e_int_field, device=dev, dtype=dtype)
        if (
            self._gas_rho2 is None
            or self._gas_rho2.shape != self.rho_field.shape
            or self._gas_rho2.device != dev
        ):
            self._gas_rho2 = torch.empty_like(self.rho_field, device=dev, dtype=dtype)
        if (
            self._gas_mom2 is None
            or self._gas_mom2.shape != self.mom_field.shape
            or self._gas_mom2.device != dev
        ):
            self._gas_mom2 = torch.empty_like(self.mom_field, device=dev, dtype=dtype)
        if (
            self._gas_e2 is None
            or self._gas_e2.shape != self.e_int_field.shape
            or self._gas_e2.device != dev
        ):
            self._gas_e2 = torch.empty_like(self.e_int_field, device=dev, dtype=dtype)
        if (
            self._gas_k1_rho is None
            or self._gas_k1_rho.shape != self.rho_field.shape
            or self._gas_k1_rho.device != dev
        ):
            self._gas_k1_rho = torch.empty_like(self.rho_field, device=dev, dtype=dtype)
        if (
            self._gas_k1_mom is None
            or self._gas_k1_mom.shape != self.mom_field.shape
            or self._gas_k1_mom.device != dev
        ):
            self._gas_k1_mom = torch.empty_like(self.mom_field, device=dev, dtype=dtype)
        if (
            self._gas_k1_e is None
            or self._gas_k1_e.shape != self.e_int_field.shape
            or self._gas_k1_e.device != dev
        ):
            self._gas_k1_e = torch.empty_like(self.e_int_field, device=dev, dtype=dtype)

    def _planck_exchange(
        self,
        *,
        heat: torch.Tensor,
        e_osc: torch.Tensor,
        omega: torch.Tensor,
        mass: torch.Tensor,
        dt: float,
        dx: float,
        c_v: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Conservative local thermal↔oscillator exchange (Planck relaxation).

        Implements `paper/main.tex` §Thermal--Oscillator Equilibration:
          - Equilibrium: E_eq = (ħ ω)/(exp(ħω/(k_B T)) - 1)
          - Update: ΔE = α (E_eq - E),  E←E+ΔE,  Q←Q-ΔE
          - Rate:   τ = (m c_v)/(4π κ r), α = 1 - exp(-dt/τ)

        [NOTES]
        - We operate in the same nondimensional convention as the ω-field code path:
          ħ_eff = 1 and k_B = 1, so E_eq = ω/(exp(ω/T)-1).
        - The only bounds enforced are physical availability: Q ≥ 0 and E_osc ≥ 0.
          This is not “clamping for stability”; it is enforcing non-negative energies
          in a conservative exchange.
        """
        if heat.numel() == 0:
            return heat, e_osc

        # Temperature from thermal store: T = Q/(m c_v)
        m = mass.to(torch.float32)
        Q = heat.to(torch.float32)
        E = e_osc.to(torch.float32)
        w = omega.to(torch.float32)

        cv = float(c_v)
        if not (cv > 0.0) or not math.isfinite(cv):
            return heat, e_osc

        denom = m * float(cv)
        # Vacuum/undefined masses → no exchange (T treated as 0).
        T = torch.where(denom > 0.0, Q / denom, torch.zeros_like(Q))

        # Planck equilibrium energy (ħ_eff=k_B=1): E_eq = ω/(exp(ω/T)-1)
        # Robust evaluation:
        # - small x=ω/T → E_eq ≈ T (equipartition)
        # - large x → E_eq ≈ ω exp(-x)
        epsT = 1e-20
        x = w / torch.clamp(T, min=epsT)
        x = torch.nan_to_num(x, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        x_small = x < 1e-4
        x_large = x > 50.0
        expx = torch.exp(torch.clamp(x, max=80.0))
        E_eq = torch.where(
            x_small,
            T,  # limit ω/(exp(ω/T)-1) → T
            torch.where(
                x_large,
                w * torch.exp(-x),  # freeze-out tail
                w / (expx - 1.0),
            ),
        )
        E_eq = torch.nan_to_num(E_eq, nan=0.0, posinf=0.0, neginf=0.0)
        E_eq = torch.clamp(E_eq, min=0.0)

        # Exchange rate α from conduction timescale.
        kappa = float(self._thermal_conductivity())
        r = 0.5 * float(dx)  # derived length scale (half-cell radius)
        if not (kappa > 0.0) or not (r > 0.0) or not (dt > 0.0):
            return heat, e_osc

        tau = (m * float(cv)) / (4.0 * math.pi * float(kappa) * float(r))
        tau = torch.where(
            torch.isfinite(tau) & (tau > 0.0), tau, torch.full_like(tau, float("inf"))
        )
        alpha = 1.0 - torch.exp(-float(dt) / tau)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = torch.clamp(alpha, 0.0, 1.0)

        dE = alpha * (E_eq - E)

        # Enforce physical availability (conservative bounds):
        # - If dE > 0, oscillator absorbs energy from heat: require Q >= dE.
        # - If dE < 0, oscillator releases energy to heat: require E >= -dE.
        dE = torch.where(dE > Q, Q, dE)
        dE = torch.where(-dE > E, -E, dE)

        E2 = E + dE
        Q2 = Q - dE

        # Fail loudly if the conservative exchange violated invariants.
        if (not torch.isfinite(E2).all()) or (not torch.isfinite(Q2).all()):
            raise RuntimeError("Planck exchange produced non-finite energies.")
        if (E2 < 0.0).any() or (Q2 < 0.0).any():
            raise RuntimeError(
                "Planck exchange violated non-negativity (E_osc or heat < 0)."
            )

        return Q2.to(heat.dtype), E2.to(e_osc.dtype)

    def _gas_primitives(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (u, p, T) from current conserved grid state."""
        # [CHOICE] primitives from (rho, mom, e_int)
        # [FORMULA] u = mom / rho
        #          p = (γ - 1) * e_int
        #          T = p / (rho * R)
        # [REASON] our grid carries internal energy density (not total energy).
        rho = self.rho_field.to(torch.float32)
        mom = self.mom_field.to(torch.float32)
        e_int = self.e_int_field.to(torch.float32)
        g = float(self.gamma)
        R_spec = float(self._R_specific())
        rho_safe = torch.where(rho > 0.0, rho, torch.ones_like(rho))
        u = mom / rho_safe[..., None]
        p = (g - 1.0) * e_int
        T = torch.zeros_like(p)
        if R_spec > 0.0:
            T = p / (rho_safe * float(R_spec))
        return u.to(self.dtype), p.to(self.dtype), T.to(self.dtype)

    def pic_scatter_conserved(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
        energies: "Tensor",
    ) -> None:
        """Conservatively scatter particle state into (rho, mom, E) grid fields."""
        gx, gy, gz = self.grid_dims
        dx = float(self.grid_spacing)

        # IMPORTANT: clear grid buffers before scatter.
        # GPU buffers are not guaranteed to be zero-initialized; accumulating into stale
        # memory can produce ~FLT_MAX garbage and poison gather (seen as E≈1e38).
        self.rho_field.zero_()
        self.mom_field.zero_()
        self.e_int_field.zero_()

        if positions.numel() == 0:
            return

        # [CHOICE] scatter backend
        # [FORMULA] "metal": sort-by-cell + Metal CIC atomics (fast)
        #          "torch": build stencils + scatter_add_ (portable, slower on MPS)
        # [REASON] MPS scatter_add is functional but not always optimized; Metal is the
        #          intended hot path. Torch remains as a debugging fallback.
        if _PIC_SCATTER_BACKEND != "torch":
            self._scatter_particles_sorted(
                positions, velocities, masses, heats, energies
            )
            return

        # Torch CIC fallback (debug/portability).
        st = cic_stencil_periodic(positions, grid_dims=(gx, gy, gz), dx=dx)

        # Dual-energy semantics: e_int_field stores INTERNAL (thermal) energy density only.
        vol = float(dx) ** 3
        inv_vol = 1.0 / vol
        w = st.w * float(inv_vol)  # (N,8) per-volume weights
        idx_flat = st.idx.reshape(-1)  # (N*8,)

        m = masses.to(torch.float32)
        v = velocities.to(torch.float32)
        q = heats.to(torch.float32)  # thermal energy per particle

        rho_contrib = (m[:, None] * w).reshape(-1)
        mom_contrib = (m[:, None, None] * v[:, None, :] * w[..., None]).reshape(-1, 3)
        E_contrib = (q[:, None] * w).reshape(-1)

        self.rho_field.view(-1).scatter_add_(0, idx_flat, rho_contrib)
        self.e_int_field.view(-1).scatter_add_(0, idx_flat, E_contrib)
        self.mom_field.view(-1, 3).scatter_add_(
            0, idx_flat[:, None].expand(-1, 3), mom_contrib
        )

    def pic_gather_primitives(
        self,
        positions: "Tensor",
        *,
        u_field: "Tensor",
        T_field: "Tensor",
    ) -> tuple["Tensor", "Tensor"]:
        """Gather (u, T) at particle positions from grid primitive fields."""
        gx, gy, gz = self.grid_dims
        dx = float(self.grid_spacing)
        if positions.numel() == 0:
            return (
                torch.empty((0, 3), device=positions.device, dtype=self.dtype),
                torch.empty((0,), device=positions.device, dtype=self.dtype),
            )
        st = cic_stencil_periodic(positions, grid_dims=(gx, gy, gz), dx=dx)
        u = gather_trilinear_vec3(st, u_field).to(self.dtype)
        T = gather_trilinear(st, T_field).to(self.dtype)
        return u, T

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

    def done_thinking(self, **state) -> bool:
        """Return True when the spatial system is sufficiently settled.

        The original "settled" heuristic was *too strict* (it effectively required
        max particle speed <= sqrt(eps), which is rarely achieved once we
        have persistent drive and finite diffusion). In practice, experiments need
        a practical definition that can stop early when dynamics are quiet.

        New policy:
        - **Early-out**: if a short window is "quiet" in a dimensionless sense
          (low CFL-like displacement, low kinetic/internal ratio, stable dt without
          RK2 rejections, and converged ω-field updates).
        """
        if not hasattr(self, "_done_thinking_step_budget"):
            # Keep this policy self-contained. Experiments should not require
            # environment variables to terminate deterministically.
            self._done_thinking_window = 50
            self._done_thinking_max_cfl = 0.02
            self._done_thinking_max_ke_frac = 1e-4
            # Relative ω-field convergence threshold (derived from float32 precision; no knobs).
            # Stop when max(||ΔΨ||/(||Ψ||+eps)) over the window is small.
            eps = float(torch.finfo(torch.float32).eps)
            self._done_thinking_max_psi_delta_rel = float(8.0 * math.sqrt(eps))
            self._done_thinking_cfl_hist = []
            self._done_thinking_ke_hist = []
            self._done_thinking_psi_hist = []
            self._done_thinking_halvings_hist = []

        velocities = state.get("velocities")
        if velocities is None or velocities.numel() == 0:
            return True

        dt_eff = float(state.get("dt", self.last_dt))
        dx = float(self.grid_spacing)

        v = velocities.to(torch.float32)
        m = state["masses"].to(torch.float32)
        q = state["heats"].to(torch.float32)
        e = state["energies"].to(torch.float32)

        # (1) displacement per step small relative to grid resolution (dimensionless).
        max_speed = float(torch.linalg.vector_norm(v, dim=1).max().detach().item())
        max_cfl = float((max_speed * dt_eff) / max(dx, 1e-12))

        # (2) kinetic energy negligible relative to internal energy.
        v2 = (v * v).sum(dim=1)
        ke_sum = float((0.5 * m * v2).sum().detach().item())
        u_sum = float((q + e).sum().detach().item())
        if u_sum <= 0.0:
            ke_frac = 0.0 if ke_sum == 0.0 else float("inf")
        else:
            ke_frac = float(ke_sum / u_sum)

        # ------------------------------------------------------------------
        # (3) Early-out window: "quiet" for several steps in a row.
        # ------------------------------------------------------------------
        w = int(self._done_thinking_window)
        if w <= 0:
            return False

        cfl_hist = self._done_thinking_cfl_hist
        ke_hist = self._done_thinking_ke_hist
        psi_hist = self._done_thinking_psi_hist
        halvings_hist = self._done_thinking_halvings_hist

        # dt-halving stability (reject-and-retry should have stopped firing).
        dt_halvings = state.get("dt_halvings", 0)
        try:
            dt_halvings_i = int(dt_halvings)
        except Exception:
            dt_halvings_i = 0

        # ω-field convergence (relative) if provided by OmegaWaveDomain.
        psi_delta_rel = state.get("psi_delta_rel", None)
        psi_delta_rel_f: float | None
        if isinstance(psi_delta_rel, (float, int)):
            psi_delta_rel_f = float(psi_delta_rel)
        else:
            psi_delta_rel_f = None

        cfl_hist.append(max_cfl)
        ke_hist.append(ke_frac)
        halvings_hist.append(dt_halvings_i)
        if psi_delta_rel_f is None:
            # Don't mix old ψ convergence values into the decision.
            psi_hist.clear()
        else:
            psi_hist.append(psi_delta_rel_f)
        if len(cfl_hist) > w:
            del cfl_hist[:-w]
        if len(ke_hist) > w:
            del ke_hist[:-w]
        if len(halvings_hist) > w:
            del halvings_hist[:-w]
        if len(psi_hist) > w:
            del psi_hist[:-w]

        if len(cfl_hist) < w or len(ke_hist) < w or len(halvings_hist) < w:
            return False

        max_cfl_tol = float(self._done_thinking_max_cfl)
        max_ke_tol = float(self._done_thinking_max_ke_frac)
        dt_ok = max(halvings_hist) == 0

        # Only require ω-field convergence if this run is producing the signal.
        if psi_delta_rel_f is None:
            psi_ok = True
        else:
            if len(psi_hist) < w:
                return False
            psi_ok = max(psi_hist) <= float(self._done_thinking_max_psi_delta_rel)

        return (
            (max(cfl_hist) <= max_cfl_tol)
            and (max(ke_hist) <= max_ke_tol)
            and dt_ok
            and psi_ok
        )

    def _ensure_spatial_hash_buffers(
        self, n: int, num_cells: int
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
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
            self._hash_sorted_particle_idx = torch.empty(
                n, dtype=torch.int32, device=dev
            )
        if (
            self._hash_cell_counts is None
            or self._hash_cell_counts.numel() != num_cells
            or self._hash_cell_counts.device != dev
        ):
            self._hash_cell_counts = torch.empty(
                num_cells, dtype=torch.int32, device=dev
            )
        if (
            self._hash_cell_starts is None
            or self._hash_cell_starts.numel() != (num_cells + 1)
            or self._hash_cell_starts.device != dev
        ):
            self._hash_cell_starts = torch.empty(
                num_cells + 1, dtype=torch.int32, device=dev
            )
        if (
            self._hash_cell_offsets is None
            or self._hash_cell_offsets.numel() != num_cells
            or self._hash_cell_offsets.device != dev
        ):
            self._hash_cell_offsets = torch.empty(
                num_cells, dtype=torch.int32, device=dev
            )

        return (
            self._hash_particle_cell_idx,
            self._hash_cell_counts,
            self._hash_cell_starts,
            self._hash_sorted_particle_idx,
            self._hash_cell_offsets,
        )

    def _ensure_collision_snapshots(
        self, velocities: "Tensor", heats: "Tensor"
    ) -> tuple["Tensor", "Tensor"]:
        """Allocate/reuse snapshot buffers and copy current state into them."""
        if (
            self._vel_in is None
            or self._vel_in.shape != velocities.shape
            or self._vel_in.device != velocities.device
        ):
            self._vel_in = torch.empty_like(velocities)
        if (
            self._heat_in is None
            or self._heat_in.shape != heats.shape
            or self._heat_in.device != heats.device
        ):
            self._heat_in = torch.empty_like(heats)
        self._vel_in.copy_(velocities)
        self._heat_in.copy_(heats)
        return self._vel_in, self._heat_in

    def _ensure_fft_poisson_cache(self) -> None:
        """Precompute 1/k^2 grid for FFT Poisson on the current device."""
        if self._fft_inv_k2 is not None:
            return
        gx, gy, gz = self.grid_dims
        h = float(self.grid_spacing)
        two_pi = float(2.0 * math.pi)

        # Wave numbers in radians per unit length.
        kx = two_pi * torch.fft.fftfreq(
            gx, d=h, device=self.device, dtype=torch.float32
        )
        ky = two_pi * torch.fft.fftfreq(
            gy, d=h, device=self.device, dtype=torch.float32
        )
        kz = two_pi * torch.fft.fftfreq(
            gz, d=h, device=self.device, dtype=torch.float32
        )

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = (KX * KX + KY * KY + KZ * KZ).to(torch.float32)
        inv_k2 = torch.zeros_like(k2)
        inv_k2 = torch.where(k2 > 0, 1.0 / k2, inv_k2)  # k=0 -> 0 (gauge)
        self._fft_k2 = k2
        self._fft_inv_k2 = inv_k2

    @property
    def ops(self):
        return self._ops

    def _ensure_sort_scatter_buffers(self, n: int, num_cells: int) -> None:
        """Allocate/reuse sort-scatter scratch buffers on the correct device."""
        dev = self.device
        # Cell index per particle
        if (
            self._sort_particle_cell_idx is None
            or self._sort_particle_cell_idx.numel() != n
        ):
            self._sort_particle_cell_idx = torch.empty(n, dtype=torch.int32, device=dev)
        # Cell counts (must be zeroed each call)
        if (
            self._sort_cell_counts is None
            or self._sort_cell_counts.numel() != num_cells
        ):
            self._sort_cell_counts = torch.empty(
                num_cells, dtype=torch.int32, device=dev
            )
        # Cell starts (prefix sum result)
        if (
            self._sort_cell_starts is None
            or self._sort_cell_starts.numel() != num_cells
        ):
            self._sort_cell_starts = torch.empty(
                num_cells, dtype=torch.int32, device=dev
            )
        # Cell offsets (working copy for atomic increment)
        if (
            self._sort_cell_offsets is None
            or self._sort_cell_offsets.numel() != num_cells
        ):
            self._sort_cell_offsets = torch.empty(
                num_cells, dtype=torch.int32, device=dev
            )
        # Sorted particle data
        if self._sort_pos_out is None or self._sort_pos_out.shape[0] != n:
            self._sort_pos_out = torch.empty(n, 3, dtype=torch.float32, device=dev)
        if self._sort_vel_out is None or self._sort_vel_out.shape[0] != n:
            self._sort_vel_out = torch.empty(n, 3, dtype=torch.float32, device=dev)
        if self._sort_mass_out is None or self._sort_mass_out.numel() != n:
            self._sort_mass_out = torch.empty(n, dtype=torch.float32, device=dev)
        if self._sort_heat_out is None or self._sort_heat_out.numel() != n:
            self._sort_heat_out = torch.empty(n, dtype=torch.float32, device=dev)
        if self._sort_energy_out is None or self._sort_energy_out.numel() != n:
            self._sort_energy_out = torch.empty(n, dtype=torch.float32, device=dev)
        if self._sort_original_idx is None or self._sort_original_idx.numel() != n:
            self._sort_original_idx = torch.empty(n, dtype=torch.int32, device=dev)

    def _scatter_particles_sorted(
        self,
        pos: "Tensor",
        vel: "Tensor",
        mass: "Tensor",
        heat: "Tensor",
        energy: "Tensor",
    ) -> None:
        """Sort-based scatter: pre-sort particles by cell, then scatter in sorted order.

        This eliminates hash-collision fallback and provides deterministic performance
        regardless of particle density.
        """
        gx, gy, gz = self.grid_dims
        num_cells = gx * gy * gz
        n = pos.shape[0]
        spacing = float(self.grid_spacing)

        self._ensure_sort_scatter_buffers(n, num_cells)

        assert self._sort_particle_cell_idx is not None
        assert self._sort_cell_counts is not None
        assert self._sort_cell_starts is not None
        assert self._sort_cell_offsets is not None
        assert self._sort_pos_out is not None
        assert self._sort_vel_out is not None
        assert self._sort_mass_out is not None
        assert self._sort_heat_out is not None
        assert self._sort_energy_out is not None
        assert self._sort_original_idx is not None

        particle_cell_idx = self._sort_particle_cell_idx
        cell_counts = self._sort_cell_counts
        cell_starts = self._sort_cell_starts
        cell_offsets = self._sort_cell_offsets
        pos_out = self._sort_pos_out
        vel_out = self._sort_vel_out
        mass_out = self._sort_mass_out
        heat_out = self._sort_heat_out
        energy_out = self._sort_energy_out
        original_idx_out = self._sort_original_idx

        # Step 1: Compute cell index for each particle
        self.ops.scatter_compute_cell_idx(
            pos,
            particle_cell_idx,
            int(gx),
            int(gy),
            int(gz),
            spacing,
        )

        # Step 2: Count particles per cell
        cell_counts.zero_()
        self.ops.scatter_count_cells(
            particle_cell_idx,
            cell_counts,
            int(gx),
            int(gy),
            int(gz),
            spacing,
        )

        # Sanity: the histogram must account for every particle.
        # If it doesn't, the reorder step will leave holes (uninitialized garbage),
        # which then gets scattered into the grid as ~1e35 energy densities.
        total = int(cell_counts.to(torch.int64).sum().detach().item())
        if total != int(n):
            raise RuntimeError(
                f"Sort-scatter cell histogram mismatch: counted={total} n={int(n)}"
            )

        # Step 3: Prefix sum to get cell_starts (using existing scan infrastructure)
        # We can use the existing exclusive_scan_u32 functions
        cell_starts.copy_(cell_counts)
        # In-place exclusive prefix sum via cumsum then shift
        torch.cumsum(cell_starts, dim=0, out=cell_starts)
        # Shift right by 1 (exclusive scan)
        cell_starts[1:] = cell_starts[:-1].clone()
        cell_starts[0] = 0

        # Step 4: Reorder particles to sorted positions
        # IMPORTANT: the Metal reorder kernel computes `dest = cell_starts[cell] + offset`,
        # where `offset = atomic_fetch_add(cell_offsets[cell], 1)`.
        # Therefore `cell_offsets` must start at 0 for every cell (offset-within-cell),
        # not at `cell_starts` (absolute pointer), otherwise we double-add the base and
        # write out of bounds.
        cell_offsets.zero_()
        self.ops.scatter_reorder_particles(
            pos,
            vel,
            mass,
            heat,
            energy,
            particle_cell_idx,
            cell_starts,
            cell_offsets,
            pos_out,
            vel_out,
            mass_out,
            heat_out,
            energy_out,
            original_idx_out,
            int(gx),
            int(gy),
            int(gz),
            spacing,
        )

        # Step 5: Scatter from sorted particles
        self.ops.scatter_sorted(
            pos_out,
            vel_out,
            mass_out,
            heat_out,
            energy_out,
            self.rho_field,
            self.mom_field,
            self.e_int_field,
            int(gx),
            int(gy),
            int(gz),
            spacing,
        )

    def solve_gravity(self) -> None:
        """Solve Poisson equation for gravitational potential.

        Poisson equation: ∇²φ = 4πGρ
        We use G from CODATA (simulation SI units) as the gravitational constant.
        """
        const = self.constants
        # [CHOICE] gravity field solve (periodic Poisson via FFT)
        # [FORMULA] ∇²φ = 4πGρ
        #          φ̂(k) = -(4πG/k²) ρ̂(k), with φ̂(0)=0
        # [REASON] exact global solve for periodic domains; avoids slow Jacobi iteration
        # [NOTES] gauge choice: subtract mean(ρ) so k=0 mode is zero (well-defined).
        #         IMPORTANT: if φ includes G via the Poisson solve, then the particle
        #         acceleration should be a = -∇φ (no additional G factor later).
        #         The Metal PIC gather applies a = -∇φ directly (no extra G factor).
        self._ensure_fft_poisson_cache()
        assert self._fft_inv_k2 is not None
        # Single-model semantics: gravity source is the gas density field ρ.
        rho = self.rho_field.to(torch.float32)
        rho = rho - rho.mean()
        rho_hat = torch.fft.fftn(rho)
        phi_hat = -(4.0 * math.pi * float(const.G)) * rho_hat * self._fft_inv_k2
        phi = torch.fft.ifftn(phi_hat).real.to(self.dtype)
        self.gravity_potential.copy_(phi)

    def _derive_dt(
        self,
        velocities: "Tensor",
        masses: "Tensor",
    ) -> float:
        # Single-model: timestep is derived from stability constraints (CFL, explicit RK2).
        _ = (velocities, masses)
        return float(self.last_dt)

    def _exclusive_scan_u32_into_starts(
        self, counts_u32_i32: "Tensor", starts_u32_i32: "Tensor", n: int
    ) -> None:
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
                t = torch.empty(
                    (size,), device=counts_u32_i32.device, dtype=torch.int32
                )
                self._scan_scratch[key] = t
            return t

        # Build level sizes: n0=n, n1=ceil(n0/tg), ... until <= tg.
        sizes: list[int] = [int(n)]
        while sizes[-1] > tg:
            sizes.append((sizes[-1] + tg - 1) // tg)

        # Per-level outputs and block sums.
        # out_levels[i] is exclusive scan of input at level i (length sizes[i]).
        out_levels: list["Tensor"] = [
            starts_u32_i32[:n]
        ]  # view: writes into `starts_u32_i32`
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
            self.ops.exclusive_scan_u32_add_block_offsets(
                out_levels[li], out_levels[li + 1], sizes[li]
            )

        # Finalize total sum into starts[n].
        self.ops.exclusive_scan_u32_finalize_total(counts_u32_i32, starts_u32_i32, n)

    def step_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Advance the coupled thermodynamic domain one timestep (periodic box).

        This executes the enforced spatial model:
          particle↔grid coupling (PIC, CIC weights) + compressible ideal-gas Navier–Stokes (RK2).
        """
        if state is None:
            state = {}
        # Normalize particle tensors onto MPS fp32 contiguous buffers.
        pos = state["positions"].to(device=self.device, dtype=self.dtype).contiguous()
        vel = state["velocities"].to(device=self.device, dtype=self.dtype).contiguous()
        heat = state["heats"].to(device=self.device, dtype=self.dtype).contiguous()
        mass = state["masses"].to(device=self.device, dtype=self.dtype).contiguous()
        # Oscillator energy store (E_i).
        e_mode_in = state.get("energies", state.get("energy_osc", None))
        if e_mode_in is None:
            raise KeyError(
                "Missing oscillator energy: expected `energies` or `energy_osc`."
            )
        e_mode = e_mode_in.to(device=self.device, dtype=self.dtype).contiguous()
        # Intrinsic frequency ω_i.
        exc_in = state.get("excitations", state.get("omega", None))
        if exc_in is None:
            raise KeyError(
                "Missing intrinsic frequency: expected `excitations` or `omega`."
            )
        exc = exc_in.to(device=self.device, dtype=self.dtype).contiguous()

        gx, gy, gz = self.grid_dims
        dx = float(self.grid_spacing)

        # Reset kernel debug log (single buffer shared across kernels in this step).
        if self.kernel_log_enabled and self.kernel_log_capacity > 0:
            self._dbg_head.zero_()

        # ------------------------------------------------------------------
        # Particle → grid: conservative deposits to (rho, mom, e_int)
        # ------------------------------------------------------------------
        self.pic_scatter_conserved(pos, vel, mass, heat, e_mode)

        # ------------------------------------------------------------------
        # Solve gravity: Poisson equation ∇²φ = 4πGρ (FFT, periodic domain)
        # ------------------------------------------------------------------
        self.solve_gravity()

        # ------------------------------------------------------------------
        # Grid evolution
        # ------------------------------------------------------------------
        gamma = float(self.gamma)
        R_spec = float(self._R_specific())
        cv = float(self._c_v())
        # Derived numerical vacuum envelope (resolution-scale, not a tunable floor).
        # We scale by total particle mass in the unit-volume domain, then by fp32 eps.
        domain_vol = float(gx) * dx * float(gy) * dx * float(gz) * dx
        mass_total = (
            float(mass.to(torch.float32).sum().detach().to("cpu").item())
            if mass.numel()
            else 0.0
        )
        rho_mean = (mass_total / domain_vol) if (domain_vol > 0.0) else 0.0
        # Numerical vacuum envelope / positivity scale.
        #
        # We keep a small *absolute* floor (gas_numerics.rho_min) to prevent
        # u=mom/rho and c=sqrt(gamma p/rho) from exploding in near-empty cells.
        # The rho_mean*eps term handles cases where the mean density is extremely small.
        rho_min = max(
            float(self.gas_numerics.rho_min),
            float(rho_mean * float(torch.finfo(torch.float32).eps)),
        )
        p_min = float(self.gas_numerics.p_min)

        # [CHOICE] stability timestep = min(advective CFL, diffusion CFL)
        # [FORMULA] dt <= CFL / max( (|u|_1 + 3c) / dx ),  c = sqrt(γ p / ρ), p=(γ-1)e_int
        # [FORMULA] dt <= CFL_diff * dx^2 / max(ν, α), ν=μ/ρ, α=k/(ρ c_v)
        # [REASON] the solver includes explicit viscous/thermal diffusion terms; ignoring
        #          diffusion CFL forces the driver to "halve-until-stable", which is
        #          equivalent but opaque. Deriving both makes the policy explicit.
        rho = self.rho_field.to(torch.float32)
        mom = self.mom_field.to(torch.float32)
        e_int = self.e_int_field.to(torch.float32)

        # Numerical low-density envelope (must match Metal):
        # - treat |rho|<=rho_eps as "low density"
        # - require tiny momentum (bounded u) and bounded internal energy density
        rho_eps = float(rho_min)
        e_eps = float(4.0 * rho_eps * float(torch.finfo(torch.float32).eps))
        e_spec_max = 10.0
        e_int_max = float(e_spec_max * rho_eps)

        # FAIL-FAST invariants for the Eulerian state *before* deriving dt.
        if (not torch.isfinite(rho).all()) or (rho < -rho_eps).any():
            raise RuntimeError(
                "Gas state invalid before CFL: rho non-finite or too negative."
            )
        if not torch.isfinite(mom).all():
            raise RuntimeError("Gas state invalid before CFL: mom non-finite.")
        if not torch.isfinite(e_int).all():
            raise RuntimeError("Gas state invalid before CFL: e_int non-finite.")

        # Low-density consistency (aligns with Metal envelope).
        vac = rho.abs() <= rho_eps
        if bool(vac.any().detach().item()):
            mom_mag = torch.linalg.vector_norm(mom[vac], dim=-1)
            e_v = e_int[vac]
            if (
                (mom_mag > rho_eps).any()
                or (e_v < -e_eps).any()
                or (e_v > e_int_max).any()
            ):
                raise RuntimeError(
                    "Gas state invalid before CFL: low-density envelope violated (mom/e_int out of bounds)."
                )

        # CFL characteristic rate only defined where rho>rho_eps (else treat as vacuum).
        mask = rho > rho_eps
        u = torch.zeros_like(mom)
        u[mask] = mom[mask] / rho[mask][..., None]
        if (e_int[mask] < 0.0).any():
            raise RuntimeError(
                "Gas state invalid before CFL: negative e_int outside vacuum envelope."
            )
        p_gas = (gamma - 1.0) * e_int
        c = torch.zeros_like(rho)
        c[mask] = torch.sqrt((gamma * p_gas[mask]) / rho[mask])
        rate = torch.zeros_like(rho)
        rate[mask] = (u[mask].abs().sum(dim=-1) + 3.0 * c[mask]) / float(dx)

        if not torch.isfinite(rate).all():
            # This is the exact "masked hot pixel" bug: do not continue.
            raise RuntimeError("CFL rate non-finite (hot pixel / vacuum singularity).")

        max_rate = float(rate.max().detach().to("cpu").item()) if rate.numel() else 0.0
        any_bad = False
        if max_rate > 0.0 and math.isfinite(max_rate):
            dt_adv = float(self.gas_numerics.cfl) / max_rate
        else:
            # If the system is fully static/vacuum, dt defaults to dx (handled below).
            dt_adv = float(dx)

        # Diffusion stability (explicit viscosity + thermal conduction).
        mu = float(self.dynamic_viscosity)
        k_thermal = float(self._thermal_conductivity())
        dt_diff = float("inf")
        if bool(mask.any().detach().item()):
            rho_m = rho[mask]  # guaranteed > rho_eps
            # ν = μ/ρ  (kinematic viscosity), α = k/(ρ c_v) (thermal diffusivity)
            nu_max = float((mu / rho_m).max().detach().to("cpu").item())
            alpha_max = float(
                (k_thermal / (rho_m * float(cv))).max().detach().to("cpu").item()
            )
            diff_max = max(nu_max, alpha_max)
            if diff_max > 0.0 and math.isfinite(diff_max):
                dt_diff = (
                    float(self.gas_numerics.cfl_diffusion)
                    * float(dx)
                    * float(dx)
                    / float(diff_max)
                )
        else:
            # Vacuum/static: diffusion does not constrain dt.
            dt_diff = float(dx)

        dt_derived = float(min(dt_adv, dt_diff, float(self.dt_max)))
        # Numerical policy (explicit): dt must be finite and positive.
        # If you want a hard ceiling, make it a model choice, not a silent clamp.
        if not (math.isfinite(float(dt_derived)) and float(dt_derived) > 0.0):
            raise RuntimeError(f"Derived dt is invalid: dt={dt_derived}")

        # Lightweight CFL diagnostics (no physics impact).
        self.last_energy_report = {
            "cfl_any_bad_rate": float(1.0 if any_bad else 0.0),
            "cfl_max_rate": float(max_rate),
            "dt_adv": float(dt_adv),
            "dt_diff": float(dt_diff),
            "dt_derived": float(dt_derived),
        }

        # Run the Eulerian grid update on-GPU (Metal kernels).
        self._ensure_gas_rk2_scratch()
        assert self._gas_rho1 is not None
        assert self._gas_mom1 is not None
        assert self._gas_e1 is not None
        assert self._gas_rho2 is not None
        assert self._gas_mom2 is not None
        assert self._gas_e2 is not None
        assert self._gas_k1_rho is not None
        assert self._gas_k1_mom is not None
        assert self._gas_k1_e is not None

        # ------------------------------------------------------------------
        # RK2 step rejection: if the solver produces inadmissible states,
        # re-run with a smaller dt (no clamping/projection of state).
        # ------------------------------------------------------------------
        dt = float(dt_derived)
        halvings = 0
        max_halvings = 10
        reject_trace: list[str] = []
        while True:
            if self.kernel_log_enabled and self.kernel_log_capacity > 0:
                self._dbg_head.zero_()

            self.ops.gas_rk2_stage1(
                self.rho_field,
                self.mom_field,
                self.e_int_field,
                self._gas_rho1,
                self._gas_mom1,
                self._gas_e1,
                self._gas_k1_rho,
                self._gas_k1_mom,
                self._gas_k1_e,
                self._dbg_head,
                self._dbg_words,
                int(self.kernel_log_capacity if self.kernel_log_enabled else 0),
                int(gx),
                int(gy),
                int(gz),
                float(dx),
                float(dt),
                float(gamma),
                float(cv),
                float(rho_min),
                float(p_min),
                float(mu),
                float(k_thermal),
            )
            self.ops.gas_rk2_stage2(
                self.rho_field,
                self.mom_field,
                self.e_int_field,
                self._gas_rho1,
                self._gas_mom1,
                self._gas_e1,
                self._gas_k1_rho,
                self._gas_k1_mom,
                self._gas_k1_e,
                self._gas_rho2,
                self._gas_mom2,
                self._gas_e2,
                self._dbg_head,
                self._dbg_words,
                int(self.kernel_log_capacity if self.kernel_log_enabled else 0),
                int(gx),
                int(gy),
                int(gz),
                float(dx),
                float(dt),
                float(gamma),
                float(cv),
                float(rho_min),
                float(p_min),
                float(mu),
                float(k_thermal),
            )

            # If the kernel marked any inadmissible cells, it poisoned outputs with NaNs.
            ok = bool(torch.isfinite(self._gas_rho2).all().detach().item())
            ok = ok and bool(torch.isfinite(self._gas_mom2).all().detach().item())
            ok = ok and bool(torch.isfinite(self._gas_e2).all().detach().item())
            if ok:
                break

            # Record a short, explicit rejection trace (kept small; no tensor dumps).
            reason_bits: list[str] = []
            if not bool(torch.isfinite(self._gas_rho2).all().detach().item()):
                reason_bits.append("rho2=nonfinite")
            if not bool(torch.isfinite(self._gas_mom2).all().detach().item()):
                reason_bits.append("mom2=nonfinite")
            if not bool(torch.isfinite(self._gas_e2).all().detach().item()):
                reason_bits.append("e2=nonfinite")
            kl_tail = (
                self._decode_kernel_log(tail=8).strip()
                if (self.kernel_log_enabled and self.kernel_log_capacity > 0)
                else ""
            )
            bad_line = ""
            if kl_tail:
                for ln in reversed(kl_tail.splitlines()):
                    if "bad" in ln.lower():
                        bad_line = ln.strip()
                        break
            if bad_line:
                reason_bits.append(bad_line)
            reason = " | ".join(reason_bits) if reason_bits else "inadmissible_grid"
            if len(reject_trace) < 8:
                reject_trace.append(f"reject dt={dt:.6g} -> {dt * 0.5:.6g} ({reason})")

            if halvings >= max_halvings:
                kernel_log = self._decode_kernel_log()
                raise RuntimeError(
                    "Gas RK2 failed after dt halving retries.\n"
                    f"- dt_derived: {float(dt_derived)}\n"
                    f"- dt_final: {float(dt)}\n"
                    f"- halvings: {int(halvings)}\n"
                    + (
                        f"\n--- kernel_log (tail) ---\n{kernel_log}\n"
                        if kernel_log
                        else ""
                    )
                )

            dt *= 0.5
            halvings += 1

        # Record the dt that actually advanced the grid.
        self.last_dt = float(dt)
        self.last_energy_report["dt_used"] = float(dt)
        self.last_energy_report["dt_halvings"] = int(halvings)
        if reject_trace:
            self.last_energy_report["rk2_reject_trace"] = "\n".join(reject_trace)

        # IMPORTANT: stage2 is stencil-based; outputs must not alias inputs.
        self.rho_field.copy_(self._gas_rho2)
        self.mom_field.copy_(self._gas_mom2)
        self.e_int_field.copy_(self._gas_e2)

        # FAIL-FAST: grid solver must not produce non-finite/negative conserved quantities.
        rho_post = self.rho_field.to(torch.float32)
        mom_post = self.mom_field.to(torch.float32)
        e_post = self.e_int_field.to(torch.float32)

        if (
            (not torch.isfinite(rho_post).all())
            or (not torch.isfinite(mom_post).all())
            or (not torch.isfinite(e_post).all())
        ):
            # Report the first offending cell, not the entire tensors.
            bad_rho = ~torch.isfinite(rho_post)
            bad_e = ~torch.isfinite(e_post)
            bad_m = ~torch.isfinite(mom_post).any(dim=-1)
            bad = bad_rho | bad_e | bad_m
            ijk = torch.nonzero(bad, as_tuple=False)
            i0, j0, k0 = (
                (int(ijk[0, 0].item()), int(ijk[0, 1].item()), int(ijk[0, 2].item()))
                if ijk.numel()
                else (-1, -1, -1)
            )
            r0 = (
                float(rho_post[i0, j0, k0].detach().to("cpu").item())
                if i0 >= 0
                else float("nan")
            )
            e0v = (
                float(e_post[i0, j0, k0].detach().to("cpu").item())
                if i0 >= 0
                else float("nan")
            )
            m0v = mom_post[i0, j0, k0].detach().to("cpu")
            mx, my, mz = (
                (float(m0v[0].item()), float(m0v[1].item()), float(m0v[2].item()))
                if i0 >= 0
                else (float("nan"),) * 3
            )
            kernel_log = self._decode_kernel_log()
            raise RuntimeError(
                "Gas RK2 produced non-finite grid state.\n"
                f"- dt: {float(dt)}\n"
                f"- first bad cell (i,j,k): ({i0},{j0},{k0})\n"
                f"- rho,e_int,mom: {r0}, {e0v}, [{mx}, {my}, {mz}]\n"
                + (f"\n--- kernel_log (tail) ---\n{kernel_log}\n" if kernel_log else "")
            )

        # Allow tiny signed rho/e_int inside the numerical vacuum envelope (see Metal admissibility).
        rho_bad = (rho_post < 0.0) & ~(rho_post.abs() <= rho_eps)
        e_bad = (e_post < 0.0) & ~(
            (rho_post.abs() <= rho_eps) & (e_post.abs() <= e_eps)
        )
        if bool(rho_bad.any().detach().item()) or bool(e_bad.any().detach().item()):
            bad = rho_bad | e_bad
            ijk = torch.nonzero(bad, as_tuple=False)
            i0, j0, k0 = (
                (int(ijk[0, 0].item()), int(ijk[0, 1].item()), int(ijk[0, 2].item()))
                if ijk.numel()
                else (-1, -1, -1)
            )
            r0 = (
                float(rho_post[i0, j0, k0].detach().to("cpu").item())
                if i0 >= 0
                else float("nan")
            )
            e0v = (
                float(e_post[i0, j0, k0].detach().to("cpu").item())
                if i0 >= 0
                else float("nan")
            )
            m0v = mom_post[i0, j0, k0].detach().to("cpu")
            mx, my, mz = (
                (float(m0v[0].item()), float(m0v[1].item()), float(m0v[2].item()))
                if i0 >= 0
                else (float("nan"),) * 3
            )
            kernel_log = self._decode_kernel_log()
            raise RuntimeError(
                "Gas RK2 produced negative rho/e_int (inadmissible).\n"
                f"- dt: {float(dt)}\n"
                f"- first bad cell (i,j,k): ({i0},{j0},{k0})\n"
                f"- rho,e_int,mom: {r0}, {e0v}, [{mx}, {my}, {mz}]\n"
                + (f"\n--- kernel_log (tail) ---\n{kernel_log}\n" if kernel_log else "")
            )

        # ------------------------------------------------------------------
        # Grid → particle: GPU-native gather + particle update (PIC)
        # ------------------------------------------------------------------
        pos_out = torch.empty_like(pos)
        vel_out = torch.empty_like(vel)
        heat_out = torch.empty_like(heat)

        domain_x = float(gx) * dx
        domain_y = float(gy) * dx
        domain_z = float(gz) * dx

        self.ops.pic_gather_update_particles(
            pos,
            mass,
            pos_out,
            vel_out,
            heat_out,
            self.rho_field,
            self.mom_field,
            self.e_int_field,
            self.gravity_potential,
            self._dbg_head,
            self._dbg_words,
            int(self.kernel_log_capacity if self.kernel_log_enabled else 0),
            int(gx),
            int(gy),
            int(gz),
            float(dx),
            float(dt),
            float(domain_x),
            float(domain_y),
            float(domain_z),
            float(gamma),
            float(R_spec),
            float(self._c_v()),
            float(self.gas_numerics.rho_min),
            float(self.gas_numerics.p_min),
            1.0,  # gravity_enabled
        )

        kernel_log = self._decode_kernel_log()

        # Fail loudly on non-physical results (no clamps).
        if (not torch.isfinite(heat_out).all()) or (heat_out < 0.0).any():
            bad_nf = ~torch.isfinite(heat_out)
            bad_neg = heat_out < 0.0
            bad = bad_nf | bad_neg
            idx = torch.nonzero(bad, as_tuple=False).flatten()
            n_bad = int(idx.numel())
            i0 = int(idx[0].item()) if n_bad else -1

            # Reconstruct the local grid state at the failing particle location.
            # This uses the same CIC stencil semantics as the kernel.
            pos0 = pos[i0 : i0 + 1]
            st0 = cic_stencil_periodic(pos0, grid_dims=self.grid_dims, dx=float(dx))
            rho0 = float(
                gather_trilinear(st0, self.rho_field.to(torch.float32))[0].item()
            )
            mom0 = gather_trilinear_vec3(st0, self.mom_field.to(torch.float32))[0]
            e_int0 = float(
                gather_trilinear(st0, self.e_int_field.to(torch.float32))[0].item()
            )
            mom0_cpu = mom0.detach().to("cpu")
            mx, my, mz = (
                float(mom0_cpu[0].item()),
                float(mom0_cpu[1].item()),
                float(mom0_cpu[2].item()),
            )
            mom2 = mx * mx + my * my + mz * mz
            ke0 = (0.5 * mom2 / rho0) if (rho0 > 0.0) else float("nan")
            m0 = float(mass[i0].detach().to("cpu").item())
            heat0 = float(heat_out[i0].detach().to("cpu").item())
            heat0_nf = bool((~torch.isfinite(heat_out[i0])).detach().to("cpu").item())
            heat0_neg = bool((heat_out[i0] < 0.0).detach().to("cpu").item())

            # Summarize global magnitudes to see whether we're overflowing.
            mom_mag = torch.linalg.vector_norm(self.mom_field.to(torch.float32), dim=-1)
            mom_max = (
                float(mom_mag.max().detach().to("cpu").item())
                if mom_mag.numel()
                else 0.0
            )
            e_int_max = (
                float(
                    self.e_int_field.to(torch.float32).max().detach().to("cpu").item()
                )
                if self.e_int_field.numel()
                else 0.0
            )
            rho_min = (
                float(self.rho_field.to(torch.float32).min().detach().to("cpu").item())
                if self.rho_field.numel()
                else 0.0
            )

            kind = "non-finite" if bool(bad_nf.any().item()) else "negative"
            raise RuntimeError(
                "PIC gather produced non-physical heat_out.\n"
                f"- kind: {kind}\n"
                f"- step dt: {float(dt)}\n"
                f"- bad count: {n_bad} / {int(heat_out.numel())}\n"
                f"- particle idx: {i0}\n"
                f"- particle mass: {m0}\n"
                f"- heat_out[i]: {heat0} (nonfinite={heat0_nf}, neg={heat0_neg})\n"
                f"- gathered rho: {rho0}\n"
                f"- gathered mom: [{mx}, {my}, {mz}] (|mom|_max grid: {mom_max})\n"
                f"- gathered e_int: {e_int0} (e_int_max grid: {e_int_max})\n"
                f"- implied ke_density: {ke0}\n"
                f"- grid rho_min: {rho_min}\n"
            )

        # ------------------------------------------------------------------
        # Local thermal ↔ oscillator exchange (Planck relaxation; conservative)
        # ------------------------------------------------------------------
        heat_x, e_mode_x = self._planck_exchange(
            heat=heat_out,
            e_osc=e_mode,
            omega=exc,
            mass=mass,
            dt=float(dt),
            dx=float(dx),
            c_v=float(self._c_v()),
        )

        # ------------------------------------------------------------------
        # Derived IDs: explicit "where|what" key (no hashing)
        # ------------------------------------------------------------------
        byte_values = state.get("byte_values", None)
        cell_morton = None
        spatial_token_ids = None
        if isinstance(byte_values, torch.Tensor):
            # [CHOICE] cell ID from current particle position (post-PIC update)
            # [FORMULA] (ix,iy,iz) = floor(pos_out / dx) mod grid_dims
            # [REASON] this is the canonical "claimed" cell at this time step; combining
            #          it with byte value yields a deterministic semantic key:
            #          spatial_token_ids = (Morton(cell) << 8) | byte.
            pos_f = pos_out.to(torch.float32)
            inv_dx = 1.0 / float(dx)
            ix = torch.floor(pos_f[:, 0] * inv_dx).to(torch.int64) % int(gx)
            iy = torch.floor(pos_f[:, 1] * inv_dx).to(torch.int64) % int(gy)
            iz = torch.floor(pos_f[:, 2] * inv_dx).to(torch.int64) % int(gz)
            cell_morton = _morton3_u32(ix, iy, iz).to(torch.int64)
            bv = byte_values.to(device=pos_out.device, dtype=torch.int64) & 0xFF
            spatial_token_ids = (cell_morton << 8) | bv

        # Return updated state (pass through unchanged fields)
        return {
            **state,
            "positions": pos_out,
            "velocities": vel_out,
            "energies": e_mode_x,  # oscillator energy
            "energy_osc": e_mode_x,
            "heats": heat_x,
            "excitations": exc,  # intrinsic ω
            "omega": exc,
            "dt": self.last_dt,
            # Exposed for observers/projectors/debugging; no physics impact.
            "cell_morton": cell_morton,
            "spatial_token_ids": spatial_token_ids,
            # Scalars for instrumentation / diagnostics (no physics impact).
            "dx": float(dx),
            "gamma": float(gamma),
            "R_specific": float(R_spec),
            "c_v": float(self._c_v()),
            "gravity_potential": self.gravity_potential,
            "kernel_log": kernel_log,
            "cfl_any_bad_rate": float(
                self.last_energy_report.get("cfl_any_bad_rate", 0.0)
            )
            if self.last_energy_report
            else 0.0,
            "cfl_max_rate": float(self.last_energy_report.get("cfl_max_rate", 0.0))
            if self.last_energy_report
            else 0.0,
            "dt_derived": float(self.last_energy_report.get("dt_derived", 0.0))
            if self.last_energy_report
            else 0.0,
            "dt_adv": float(self.last_energy_report.get("dt_adv", 0.0))
            if self.last_energy_report
            else 0.0,
            "dt_diff": float(self.last_energy_report.get("dt_diff", 0.0))
            if self.last_energy_report
            else 0.0,
            "dt_used": float(self.last_energy_report.get("dt_used", self.last_dt))
            if self.last_energy_report
            else float(self.last_dt),
            "dt_halvings": int(self.last_energy_report.get("dt_halvings", 0))
            if self.last_energy_report
            else 0,
            "rk2_reject_trace": str(self.last_energy_report.get("rk2_reject_trace", ""))
            if self.last_energy_report
            else "",
        }

    def step(self, *args: Any, **state: Any):
        return self.step_state(state)


class OmegaWaveDomain:
    r"""Metal-accelerated ω-field domain Ψ(ω) (complex ω-lattice wave dynamics).

    This implements a fixed ω-lattice (uniform bins ω_k) carrying a complex field
    (Psi_k = Psi(omega_k)) evolved by a dissipative Gross–Pitaevskii / NLSE-style
    update in ω-index space. Particles couple into the field through:

    - a frequency lineshape (Lorentzian) parameterized by a per-mode linewidth (gamma_k)
    - a real-space overlap proxy via sparse per-mode anchors

    The public interface intentionally exposes **no config surface**: the only
    degree of freedom is `grid_size`. Everything else is either a universal
    constant, a fixed model choice, or derived from `grid_size` / observed data.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (64, 64, 64),
        *,
        num_modes: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")

        self.grid_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        self.device = torch.device("mps")
        self.dtype = torch.float32

        self.grid_spacing: float = _derived_grid_spacing(self.grid_size)
        gx, gy, gz = self.grid_size
        self.domain_x = float(gx) * self.grid_spacing
        self.domain_y = float(gy) * self.grid_spacing
        self.domain_z = float(gz) * self.grid_spacing

        # Default ω-lattice size derived from grid topology.
        max_dim = int(max(self.grid_size))
        derived_modes: int = 1 << max(0, (max_dim - 1).bit_length())
        self.num_modes = int(num_modes) if num_modes is not None else int(derived_modes)
        if self.num_modes <= 0:
            raise ValueError(f"Derived num_modes must be > 0, got {self.num_modes}")

        # Compatibility surface for tests (fixed lattice, no spawning).
        self.max_carriers: int = int(self.num_modes)
        self.num_carriers: int = int(self.num_modes)

        # ω-field state at lattice sites ω_k:
        #   Ψ_k = ψ_real[k] + i ψ_imag[k]
        # plus a per-site linewidth γ_k (Lorentzian coupling width).
        self.psi_real = torch.zeros(
            self.num_modes, device=self.device, dtype=self.dtype
        )
        self.psi_imag = torch.zeros(
            self.num_modes, device=self.device, dtype=self.dtype
        )
        # Previous step buffers (for convergence signals; no physics impact).
        self._prev_psi_real: torch.Tensor | None = None
        self._prev_psi_imag: torch.Tensor | None = None
        self.omega_lattice = torch.zeros(
            self.num_modes, device=self.device, dtype=self.dtype
        )
        self.mode_linewidth = torch.ones(
            self.num_modes, device=self.device, dtype=self.dtype
        )

        # Anchors approximate overlap integrals (mechanically linked to Metal).
        self.anchor_slots: int = int(_METAL_MODE_ANCHORS)
        self.mode_anchor_idx = torch.full(
            (self.num_modes * self.anchor_slots,),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        self.mode_anchor_weight = torch.zeros(
            (self.num_modes * self.anchor_slots,), device=self.device, dtype=self.dtype
        )
        self._anchors_seeded: bool = False

        # Fixed "active mode count" snapshot (device-side scalar).
        self._num_modes_snapshot = torch.tensor(
            [self.num_modes], device=self.device, dtype=torch.int32
        )

        # Mode accumulators backing store (ModeAccumulators is 8 × 4 bytes in Metal).
        self._mode_accums = torch.zeros(
            self.num_modes * 8, device=self.device, dtype=torch.int32
        )

        # Simple ω-binning (uniform lattice → identity permutation).
        self._num_bins = int(self.num_modes)
        self._bin_starts = torch.arange(
            self._num_bins + 1, device=self.device, dtype=torch.int32
        )
        self._mode_binned_idx = torch.arange(
            self._num_bins, device=self.device, dtype=torch.int32
        )
        self._bin_params = torch.zeros((2,), device=self.device, dtype=self.dtype)

        self._omega_initialized = False
        self._inv_domega2: float = 0.0
        self._gate_width_min: float = 0.0
        self._gate_width_max: float = 0.0
        self._omega_min: float = 0.0
        self._domega: float = 0.0

        # [CHOICE] default spatial coherence length σ_x (derived)
        # [FORMULA] σ_x = 0.25 * min(domain_x, domain_y, domain_z)
        self.spatial_sigma = 0.25 * min(self.domain_x, self.domain_y, self.domain_z)

        # ------------------------------------------------------------------
        # Homeostasis: heat → work budget for coupling to Ψ(ω)
        # ------------------------------------------------------------------
        # [CHOICE] metabolic work rate
        # [FORMULA] W_req = metabolic_rate * A_i * dt
        # [REASON] Defines "work" as effort to align Ψ(ω) with particle phase.
        #          Heat pays for work; GPE energy_decay dissipates it.
        self.metabolic_rate: float = 0.5

        # [CHOICE] spectral timestep derived from spatial discretization
        # [FORMULA] Δt_ω = Δx
        self.dt = float(dt) if dt is not None else float(self.grid_spacing)

        # GPE coefficients (fixed model choice unless overridden by wrapper).
        self.hbar_eff: float = 1.0
        self.mass_eff: float = 1.0
        self.g_interaction: float = -1.0
        self.energy_decay: float = float(1.0 / float(self.num_modes))

        self._rng_seed: int = 1

        from .jit import load_manifold_metal_ops

        self._ops = load_manifold_metal_ops()

    @property
    def ops(self):
        return self._ops

    def synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def _init_omega_lattice(self, omega_min: float, omega_max: float) -> None:
        if not (omega_max > omega_min):
            raise ValueError(
                f"omega_max must be > omega_min, got {omega_max} <= {omega_min}"
            )

        self.omega_lattice.copy_(
            torch.linspace(
                omega_min,
                omega_max,
                steps=self.num_modes,
                device=self.device,
                dtype=self.dtype,
            )
        )

        if self.num_modes >= 2:
            domega = float((omega_max - omega_min) / float(self.num_modes - 1))
            inv_bin_width = (1.0 / domega) if domega != 0.0 else 0.0
            self._inv_domega2 = (1.0 / (domega * domega)) if domega != 0.0 else 0.0
        else:
            domega = 0.0
            inv_bin_width = 0.0
            self._inv_domega2 = 0.0

        # CoherenceBinParams { omega_min, inv_bin_width }
        self._bin_params[0] = float(omega_min)
        self._bin_params[1] = float(inv_bin_width)
        self._omega_min = float(omega_min)
        self._domega = float(domega)

        # [CHOICE] initialize linewidths from ω discretization (derived)
        # [FORMULA] γ_init = Δω, γ_min = 0.25Δω, γ_max = 4Δω
        if domega > 0.0:
            self.mode_linewidth.fill_(float(domega))
            self._gate_width_min = float(0.25 * domega)
            self._gate_width_max = float(4.0 * domega)
        else:
            self.mode_linewidth.fill_(1.0)
            self._gate_width_min = 1e-6
            self._gate_width_max = 1.0

        self._omega_initialized = True
        self._anchors_seeded = False

    def _seed_mode_anchors(
        self,
        *,
        particle_omega: torch.Tensor,
        particle_amp: torch.Tensor,
    ) -> None:
        """Deterministically seed mode anchors from the current particle set.

        This is not a “config/flag”; it makes the intended overlap proxy well-defined
        from step 0 so diagnostics/visualization don’t depend on rare random refresh.
        """
        M = int(self.num_modes)
        slots = int(self.anchor_slots)
        if M <= 0 or slots <= 0:
            return
        if particle_omega.numel() == 0:
            return

        omega_min = float(self._omega_min)
        domega = float(self._domega)
        if not (domega > 0.0 and math.isfinite(domega)):
            return

        # Nearest ω-bin for each particle (uniform lattice).
        f = (particle_omega.to(torch.float32) - float(omega_min)) / float(domega)
        k_idx = torch.round(f).to(torch.int64).clamp_(0, M - 1)  # (N,)

        # Clear existing anchors.
        self.mode_anchor_idx.fill_(-1)
        self.mode_anchor_weight.zero_()

        # For each mode, pick up to `slots` particles with largest amplitude.
        # N is typically small (batch_size), so a simple per-mode pass is fine and deterministic.
        amp = particle_amp.to(torch.float32)
        for k in range(M):
            mask = k_idx == int(k)
            if not bool(mask.any().detach().item()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            # Sort by amplitude descending.
            a = amp[idx]
            order = torch.argsort(a, descending=True)
            chosen = idx[order[:slots]]
            base = k * slots
            # Fill idx/weight (weight uses amplitude as a support proxy).
            self.mode_anchor_idx[base : base + chosen.numel()] = chosen.to(torch.int32)
            self.mode_anchor_weight[base : base + chosen.numel()] = amp[chosen].to(
                self.dtype
            )

    def step(self, **state) -> dict[str, Any]:
        """Advance the ω-field one timestep."""
        # Canonical wave-layer keys:
        # - phase: particle phase θ_i
        # - omega: intrinsic particle frequency ω_i
        # - energy_osc: oscillator/internal-mode energy E_i (amplitude A_i = sqrt(E_i))
        phase_in = state["phase"]
        omega_in = state["omega"]
        e_osc_in = state["energy_osc"]

        particle_phase = phase_in.to(device=self.device, dtype=self.dtype).contiguous()
        particle_omega = omega_in.to(device=self.device, dtype=self.dtype).contiguous()
        energy = e_osc_in.to(device=self.device, dtype=self.dtype).contiguous()
        particle_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()
        pos = state["positions"].to(device=self.device, dtype=self.dtype).contiguous()
        heat = state["heats"].to(device=self.device, dtype=self.dtype).contiguous()

        if not self._omega_initialized:
            omega_min = float(particle_omega.min().detach().item())
            omega_max = float(particle_omega.max().detach().item())
            if not (omega_max > omega_min):
                omega_min -= 1.0
                omega_max += 1.0
            self._init_omega_lattice(omega_min, omega_max)

        # Ensure anchors are populated deterministically (for overlap + visualization).
        if not self._anchors_seeded:
            self._seed_mode_anchors(
                particle_omega=particle_omega, particle_amp=particle_amp
            )
            self._anchors_seeded = True

        self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
        self._mode_accums.zero_()

        # Derived “floors” from floating precision (no knobs).
        eps = float(torch.finfo(torch.float32).eps)
        # Weight floor is purely a float32 denormal guard; keep it as small as possible
        # so large-N normalized runs still couple (no tuning knobs).
        weight_floor = float(1.0 * math.sqrt(eps))
        anchor_eps = float(math.sqrt(eps))

        # ------------------------------------------------------------------
        # Derived spatial coherence length σ_x(T) (dual-energy / high-Mach safe)
        # ------------------------------------------------------------------
        # [CHOICE] temperature-coupled coherence length (thermal de Broglie, nondimensional)
        # [FORMULA] T_i = Q_i / (m_i c_v)
        #          σ_x = sqrt(2π) / sqrt(m̄ * T̄)   with ħ_eff = k_B = 1
        # [REASON] makes overlap radius shrink at high temperature (decoherence) and
        #          expand in cold regimes, without introducing hand-tuned knobs.
        # [NOTES] - We clamp only to derived resolution/domain bounds:
        #           σ_min = Δx, σ_max = 0.5 * min(Lx, Ly, Lz).
        #         - If T̄ is non-finite or ≤ 0, we take σ_x = σ_max (cold/undefined bath).
        sigma_min = float(self.grid_spacing)
        sigma_max = float(0.5 * min(self.domain_x, self.domain_y, self.domain_z))
        sigma_x = float(self.spatial_sigma)
        c_v = float(state.get("c_v", 0.0))
        heats = state.get("heats", None)
        masses = state.get("masses", None)
        if (
            c_v > 0.0
            and isinstance(heats, torch.Tensor)
            and isinstance(masses, torch.Tensor)
        ):
            q = heats.to(device=self.device, dtype=self.dtype)
            m = masses.to(device=self.device, dtype=self.dtype)
            if q.numel() == m.numel() and q.numel() > 0:
                mask = m > 0.0
                if bool(mask.any().detach().item()):
                    T = q[mask] / (m[mask] * float(c_v))
                    T_mean = T.to(torch.float32).mean()
                    m_mean = m[mask].to(torch.float32).mean()
                    denom = m_mean * T_mean
                    lam = torch.tensor(
                        float(sigma_max), device=self.device, dtype=torch.float32
                    )
                    if bool((torch.isfinite(denom) & (denom > 0.0)).detach().item()):
                        lam = math.sqrt(2.0 * math.pi) / torch.sqrt(denom)
                    lam = torch.clamp(lam, min=float(sigma_min), max=float(sigma_max))
                    sigma_x = float(lam.detach().item())

        # Persist as stateful derived parameter (no config surface).
        self.spatial_sigma = sigma_x

        # Derived coupling normalization (avoid scale dependence on M).
        coupling_scale = float(1.0 / math.sqrt(float(self.num_modes)))

        gate_width_min = float(
            self._gate_width_min if self._gate_width_min > 0.0 else 1e-6
        )
        gate_width_max = float(
            self._gate_width_max
            if self._gate_width_max > gate_width_min
            else (4.0 * gate_width_min)
        )

        # GPE coefficients (fixed model choice unless overridden by wrapper).
        hbar_eff = float(self.hbar_eff)
        mass_eff = float(self.mass_eff)
        g_interaction = float(self.g_interaction)
        chemical_potential = 0.0
        energy_decay = float(self.energy_decay)

        # 1) Accumulate particle support into mode accumulators (PIC-like coupling).
        self.ops.coherence_accumulate_forces(
            particle_phase,
            particle_omega,
            particle_amp,
            pos,
            self.omega_lattice,
            self.mode_linewidth,
            self.mode_anchor_idx,
            self.mode_anchor_weight,
            self._mode_accums,
            self._bin_starts,
            self._mode_binned_idx,
            self._bin_params,
            int(self._num_bins),
            heat,
            int(particle_phase.shape[0]),
            self._num_modes_snapshot,
            int(self.num_modes),
            float(self.dt),
            float(self.metabolic_rate),
            float(gate_width_min),
            float(gate_width_max),
            float(weight_floor),
            float(self.domain_x),
            float(self.domain_y),
            float(self.domain_z),
            float(sigma_x),
        )

        # 2) Dissipative GPE step on Ψ(ω_k).
        self.ops.coherence_gpe_step(
            particle_phase,
            particle_omega,
            particle_amp,
            pos,
            self.psi_real,
            self.psi_imag,
            self.omega_lattice,
            self.mode_linewidth,
            self.mode_anchor_idx,
            self.mode_anchor_weight,
            self._mode_accums,
            self._num_modes_snapshot,
            int(self.num_modes),
            float(self.dt),
            float(hbar_eff),
            float(mass_eff),
            float(g_interaction),
            float(energy_decay),
            float(chemical_potential),
            float(self._inv_domega2),
            int(self._rng_seed) & 0xFFFFFFFF,
            float(anchor_eps),
            float(gate_width_min),
            float(gate_width_max),
            float(weight_floor),
            float(self.domain_x),
            float(self.domain_y),
            float(self.domain_z),
            float(sigma_x),
        )

        # 3) Update particle phases from the coherence field.
        self.ops.coherence_update_oscillator_phases(
            particle_phase,
            particle_omega,
            particle_amp,
            self.psi_real,
            self.psi_imag,
            self.omega_lattice,
            self.mode_linewidth,
            self.mode_anchor_idx,
            self.mode_anchor_weight,
            self._num_modes_snapshot,
            self._bin_starts,
            self._mode_binned_idx,
            self._bin_params,
            int(self._num_bins),
            int(self.num_modes),
            float(self.dt),
            float(coupling_scale),
            float(gate_width_min),
            float(gate_width_max),
            pos,
            float(self.domain_x),
            float(self.domain_y),
            float(self.domain_z),
            float(sigma_x),
        )

        # ------------------------------------------------------------------
        # Convergence signals for settling / dashboard: ||ΔΨ|| and ||ΔΨ||/||Ψ||
        # ------------------------------------------------------------------
        dpsi_rms = 0.0
        psi_rms = 0.0
        if self.psi_real.numel() > 0:
            if (
                self._prev_psi_real is None
                or self._prev_psi_imag is None
                or self._prev_psi_real.shape != self.psi_real.shape
                or self._prev_psi_real.device != self.psi_real.device
            ):
                self._prev_psi_real = self.psi_real.detach().clone()
                self._prev_psi_imag = self.psi_imag.detach().clone()
                dpsi_rms = 0.0
                psi2 = (
                    self.psi_real.to(torch.float32) * self.psi_real.to(torch.float32)
                    + self.psi_imag.to(torch.float32) * self.psi_imag.to(torch.float32)
                ).mean()
                psi_rms = float(torch.sqrt(torch.clamp(psi2, min=0.0)).detach().item())
            else:
                dr = (self.psi_real - self._prev_psi_real).to(torch.float32)
                di = (self.psi_imag - self._prev_psi_imag).to(torch.float32)
                dpsi2 = (dr * dr + di * di).mean()
                dpsi_rms = float(
                    torch.sqrt(torch.clamp(dpsi2, min=0.0)).detach().item()
                )

                pr = self.psi_real.to(torch.float32)
                pi = self.psi_imag.to(torch.float32)
                psi2 = (pr * pr + pi * pi).mean()
                psi_rms = float(torch.sqrt(torch.clamp(psi2, min=0.0)).detach().item())

                self._prev_psi_real.copy_(self.psi_real)
                self._prev_psi_imag.copy_(self.psi_imag)

        # Numerical robustness: never emit NaN/Inf for dashboard/health signals.
        # If ||Ψ|| is extremely small, treat the relative change as absolute change
        # divided by a tiny epsilon (still finite).
        den = max(float(psi_rms), 1e-12)
        psi_delta_rel = float(dpsi_rms / den)

        mr = self.psi_real
        mi = self.psi_imag
        amp = torch.sqrt(mr * mr + mi * mi)
        phase = torch.atan2(mi, mr)

        # Derived “state” annotation (no physics impact; observer convenience).
        # [CHOICE] stable/crystallized cutoffs derived from amplitude distribution
        # [FORMULA] stable = Q75(|Ψ|), crystallized = Q90(|Ψ|)
        if amp.numel() > 0:
            st_th = float(torch.quantile(amp.detach(), 0.75).item())
            cr_th = float(torch.quantile(amp.detach(), 0.90).item())
        else:
            st_th = 0.0
            cr_th = 0.0

        denom = cr_th if cr_th > 0.0 else 1.0
        conflict = (1.0 - torch.clamp(amp / denom, 0.0, 1.0)).to(self.dtype)

        mode_state = torch.zeros_like(amp, dtype=torch.int32, device=amp.device)
        if st_th > 0.0:
            mode_state = torch.where(
                amp >= st_th, torch.ones_like(mode_state), mode_state
            )
        if cr_th > 0.0:
            mode_state = torch.where(
                amp >= cr_th, torch.full_like(mode_state, 2), mode_state
            )

        # Return updated state (pass through unchanged fields)
        return {
            **state,
            "num_modes": self._num_modes_snapshot.clone(),
            "heats": heat,
            # Settling / convergence signals (no physics impact; used for done_thinking + dashboard).
            "psi_delta_rms": float(dpsi_rms),
            "psi_rms": float(psi_rms),
            "psi_delta_rel": float(psi_delta_rel),
            # Canonical ω-field outputs (preferred)
            "omega_lattice": self.omega_lattice,
            "psi_real": self.psi_real,
            "psi_imag": self.psi_imag,
            "psi_amplitude": amp,
            "psi_phase": phase,
            "mode_linewidth": self.mode_linewidth,
            "mode_conflict": conflict,
            "mode_state": mode_state,
            # Observer-friendly generic names (no legacy vocabulary)
            "frequencies": self.omega_lattice,
            "gate_widths": self.mode_linewidth,
            "amplitudes": amp,
            "phases": phase,
            "conflict": conflict,
            # Anchor state (for diagnostics / visualization; not required by physics)
            "mode_anchor_idx": self.mode_anchor_idx,
            "mode_anchor_weight": self.mode_anchor_weight,
            "spatial_sigma": float(self.spatial_sigma),
            # Canonical particle wave-state outputs (preferred)
            "phase": particle_phase,
            "omega": particle_omega,
            "energy_osc": energy,
        }


class CoherenceDomain:
    """Compatibility wrapper for coherence field unit tests.

    The main Metal implementation is `OmegaWaveDomain`. Tests expect an explicit
    config surface and a positional `step(phase, omega, energy, particle_positions=...)`.
    """

    def __init__(
        self,
        config: CoherenceFieldConfig,
        *,
        grid_size: tuple[int, int, int] = (64, 64, 64),
        dt: float = 0.01,
        device: str = "mps",
    ) -> None:
        if device != "mps":
            raise RuntimeError(
                f"CoherenceDomain (Metal) requires device='mps' (got {device!r})"
            )

        self._impl = OmegaWaveDomain(
            grid_size=grid_size,
            num_modes=int(config.omega_bins),
            dt=float(dt),
        )

        if not (config.omega_bins > 0):
            raise ValueError(f"omega_bins must be > 0, got {config.omega_bins}")
        if not (config.omega_max > config.omega_min):
            raise ValueError(
                f"omega_max must be > omega_min, got {config.omega_max} <= {config.omega_min}"
            )

        self._impl.hbar_eff = float(config.hbar_eff)
        self._impl.mass_eff = float(config.mass_eff)
        self._impl.g_interaction = float(config.g_interaction)
        self._impl.energy_decay = (
            float(config.energy_decay)
            if float(config.energy_decay) > 0.0
            else float(1.0 / float(self._impl.num_modes))
        )

        # Fixed lattice (do not reinitialize from particle ω in `step`).
        self._impl._init_omega_lattice(float(config.omega_min), float(config.omega_max))
        self._impl._omega_initialized = True

    def __getattr__(self, name: str):  # pragma: no cover
        return getattr(self._impl, name)

    def step(
        self,
        osc_phase: "Tensor",
        osc_omega: "Tensor",
        osc_energy: "Tensor",
        *,
        particle_positions: "Tensor",
    ) -> dict[str, Any]:
        N = int(osc_phase.shape[0])
        heats = torch.zeros((N,), device=self._impl.device, dtype=self._impl.dtype)
        return self._impl.step(
            phase=osc_phase,
            omega=osc_omega,
            energy_osc=osc_energy,
            positions=particle_positions,
            heats=heats,
        )
