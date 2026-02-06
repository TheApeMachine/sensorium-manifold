"""CUDA/Triton implementation of manifold physics components.

This module mirrors the API shape of `optimizer/metal/manifold_physics.py` for CUDA.
Currently implemented:
- Spectral carriers (resonance potential) with crystallization + anchored top-down bias
- Idle compute modes (consolidate / disambiguate / explore)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

import torch

from sensorium.kernels.physics_units import (
    PhysicalConstants,
    UnitSystem,
    assert_finite_constants,
    gas_R_specific_sim,
)
from sensorium.kernels.gas_dynamics import (
    GasNumerics,
    conserved_to_primitives,
    advance_navier_stokes_rk2,
    compute_dt_cfl,
    compute_dt_stable,
    central_diff_periodic,
)
from sensorium.kernels.pic import (
    cic_stencil_periodic,
    scatter_conserved_cic,
    gather_trilinear,
    gather_trilinear_vec3,
)

from . import manifold_physics_kernels as k
from . import manifold_grid_kernels as g
from . import spatial_hash_kernels as sh

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ThermodynamicsDomainConfig:
    """CUDA/Triton thermodynamics domain config (matches Metal semantics)."""

    grid_size: tuple[int, int, int] = (64, 64, 64)
    grid_spacing: float = 1.0
    dt_max: float = 0.01

    # [CHOICE] unit system (simulation units → SI units)
    # [FORMULA] x_SI = x_sim * unit_scale
    # [REASON] derive universal constants from CODATA + explicit base-unit mapping
    # [NOTES] Default is the identity mapping (1 sim unit == 1 SI unit).
    unit_system: UnitSystem = field(default_factory=UnitSystem.si)

    # Minimal ideal-gas Navier–Stokes + PIC parameters (single model).
    molecular_weight_kg_per_mol: float = 0.02897
    gamma: float = 1.4
    dynamic_viscosity: float = 1.8e-5
    prandtl: float = 0.71

    def physical_constants(self) -> PhysicalConstants:
        c = PhysicalConstants.from_codata_si(self.unit_system)
        assert_finite_constants(c)
        return c

    def R_specific(self) -> float:
        return gas_R_specific_sim(self.unit_system, molecular_weight_kg_per_mol=float(self.molecular_weight_kg_per_mol))

    def c_v(self) -> float:
        return float(self.R_specific()) / (float(self.gamma) - 1.0)

    def c_p(self) -> float:
        g = float(self.gamma)
        return (g * float(self.R_specific())) / (g - 1.0)


class ThermodynamicsDomain:
    """CUDA/Triton-accelerated thermodynamics domain (grid + particles)."""

    def __init__(self, config: ThermodynamicsDomainConfig, device: str = "cuda"):
        if device != "cuda":
            raise RuntimeError(f"ThermodynamicsDomain(CUDA) requires device='cuda', got '{device}'")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32

        gx, gy, gz = config.grid_size
        self.grid_dims = (gx, gy, gz)

        self.gravity_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.gravity_potential = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        # Total internal energy per cell (Q_cell := Σ w_i (Q_i + E_osc,i)), used to derive temperature.
        self.heat_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.temperature_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)

        # Compressible gas (ideal-gas Navier–Stokes): conserved grid state.
        self.rho_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.mom_field = torch.zeros(gx, gy, gz, 3, device=self.device, dtype=self.dtype)
        self.E_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.gas_numerics = GasNumerics()

        # FFT caches for periodic Poisson + diffusion.
        self._fft_k2: Optional[torch.Tensor] = None      # (gx, gy, gz) fp32 CUDA
        self._fft_inv_k2: Optional[torch.Tensor] = None  # (gx, gy, gz) fp32 CUDA

    def _gas_primitives(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (u, p, T) from current conserved grid state."""
        R_spec = float(self.config.R_specific())
        u, p, T, _ = conserved_to_primitives(
            self.rho_field.to(torch.float32),
            self.mom_field.to(torch.float32),
            self.E_field.to(torch.float32),
            gamma=float(self.config.gamma),
            R_specific=R_spec,
            numerics=self.gas_numerics,
        )
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
        dx = float(self.config.grid_spacing)

        self.rho_field.zero_()
        self.mom_field.zero_()
        self.E_field.zero_()

        if positions.numel() == 0:
            return

        st = cic_stencil_periodic(positions, grid_dims=(gx, gy, gz), dx=dx)
        internal_energy = heats + energies
        scatter_conserved_cic(
            st,
            grid_dims=(gx, gy, gz),
            dx=dx,
            masses=masses,
            velocities=velocities,
            internal_energy=internal_energy,
            out_rho=self.rho_field,
            out_mom=self.mom_field,
            out_E=self.E_field,
        )

    def pic_gather_primitives(
        self,
        positions: "Tensor",
        *,
        u_field: "Tensor",
        T_field: "Tensor",
    ) -> tuple["Tensor", "Tensor"]:
        gx, gy, gz = self.grid_dims
        dx = float(self.config.grid_spacing)
        if positions.numel() == 0:
            return (
                torch.empty((0, 3), device=positions.device, dtype=self.dtype),
                torch.empty((0,), device=positions.device, dtype=self.dtype),
            )
        st = cic_stencil_periodic(positions, grid_dims=(gx, gy, gz), dx=dx)
        u = gather_trilinear_vec3(st, u_field).to(self.dtype)
        T = gather_trilinear(st, T_field).to(self.dtype)
        return u, T

    def _ensure_fft_cache(self) -> None:
        """Precompute k² and 1/k² grids for periodic FFT solves on CUDA."""
        if self._fft_inv_k2 is not None:
            return
        gx, gy, gz = self.grid_dims
        h = float(self.config.grid_spacing)
        two_pi = float(2.0 * math.pi)

        kx = two_pi * torch.fft.fftfreq(gx, d=h, device=self.device, dtype=torch.float32)
        ky = two_pi * torch.fft.fftfreq(gy, d=h, device=self.device, dtype=torch.float32)
        kz = two_pi * torch.fft.fftfreq(gz, d=h, device=self.device, dtype=torch.float32)

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = (KX * KX + KY * KY + KZ * KZ).to(torch.float32)
        inv_k2 = torch.zeros_like(k2)
        inv_k2 = torch.where(k2 > 0, 1.0 / k2, inv_k2)  # k=0 -> 0 (gauge)
        self._fft_k2 = k2
        self._fft_inv_k2 = inv_k2

    def scatter_particles(
        self,
        positions: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
        energies: "Tensor",
    ) -> None:
        raise RuntimeError(
            "scatter_particles() belonged to the legacy particle-field model. "
            "The single spatial model is now compressible ideal-gas Navier–Stokes + PIC."
        )

    def solve_gravity(self) -> None:
        """Solve periodic Poisson equation for gravitational potential via FFT.

        [FORMULA] ∇²φ = 4πGρ
                 φ̂(k) = -(4πG/k²) ρ̂(k), with φ̂(0)=0
        [NOTES] Single-model semantics: ρ is the gas density field (`rho_field`).
               Subtract mean density so the k=0 mode is well-defined (gauge).
        """
        cfg = self.config
        const = cfg.physical_constants()
        self._ensure_fft_cache()
        assert self._fft_inv_k2 is not None
        rho = self.rho_field.to(torch.float32)
        rho = rho - rho.mean()
        rho_hat = torch.fft.fftn(rho)
        phi_hat = -(4.0 * math.pi * float(const.G)) * rho_hat * self._fft_inv_k2
        phi = torch.fft.ifftn(phi_hat).real.to(self.dtype)
        self.gravity_potential.copy_(phi)

    def diffuse_heat(self) -> None:
        raise RuntimeError(
            "diffuse_heat() belonged to the legacy temperature-field model. "
            "The single spatial model is now compressible ideal-gas Navier–Stokes + PIC."
        )

    def scale_report(
        self,
        *,
        velocities: "Tensor",
        masses: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        dt: Optional[float] = None,
    ) -> dict[str, float]:
        raise RuntimeError(
            "scale_report() was part of the legacy particle-field model. "
            "This codebase now implements a single spatial model: compressible ideal-gas Navier–Stokes + PIC."
        )

    def gather_update_particles(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        raise RuntimeError(
            "gather_update_particles() belonged to the legacy particle-field model. "
            "The single spatial model is now compressible ideal-gas Navier–Stokes + PIC."
        )

    def step(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        cfg = self.config
        dx = float(cfg.grid_spacing)
        dt_cfg = float(cfg.dt_max)

        # Single model: compressible ideal-gas Navier–Stokes + PIC (periodic domain).
        # 1) Particle → grid (conserved)
        self.pic_scatter_conserved(positions, velocities, masses, heats, energies)

        # 2) Gravity solve from gas density ρ
        self.solve_gravity()
        phi = self.gravity_potential.to(torch.float32)
        g_accel = torch.stack(
            [
                -central_diff_periodic(phi, dx, 0),
                -central_diff_periodic(phi, dx, 1),
                -central_diff_periodic(phi, dx, 2),
            ],
            dim=-1,
        ).to(torch.float32)

        mu = float(cfg.dynamic_viscosity)
        Pr = float(cfg.prandtl)
        if Pr <= 0.0:
            raise ValueError(f"ThermodynamicsDomainConfig.prandtl must be > 0 (got {Pr})")
        k_thermal = mu * float(cfg.c_p()) / Pr

        # 3) dt cap by advective + diffusive constraints
        dt_stable = compute_dt_stable(
            self.rho_field.to(torch.float32),
            self.mom_field.to(torch.float32),
            self.E_field.to(torch.float32),
            gamma=float(cfg.gamma),
            R_specific=float(cfg.R_specific()),
            dx=dx,
            mu=mu,
            k_thermal=k_thermal,
            c_p=float(cfg.c_p()),
            numerics=self.gas_numerics,
        )
        dt = min(dt_cfg, float(dt_stable)) if (math.isfinite(float(dt_stable)) and float(dt_stable) > 0.0) else dt_cfg

        rho_n, mom_n, E_n = advance_navier_stokes_rk2(
            self.rho_field.to(torch.float32),
            self.mom_field.to(torch.float32),
            self.E_field.to(torch.float32),
            gamma=float(cfg.gamma),
            R_specific=float(cfg.R_specific()),
            dx=dx,
            dt=float(dt),
            mu=mu,
            k_thermal=k_thermal,
            g_accel=g_accel,
            numerics=self.gas_numerics,
        )
        self.rho_field.copy_(rho_n.to(self.dtype))
        self.mom_field.copy_(mom_n.to(self.dtype))
        self.E_field.copy_(E_n.to(self.dtype))

        u_field, _p_field, T_field = self._gas_primitives()
        u_p, T_p = self.pic_gather_primitives(positions, u_field=u_field, T_field=T_field)

        velocities = u_p
        positions = (positions + velocities * float(dt)).contiguous()
        domain = torch.tensor(
            [self.grid_dims[0] * dx, self.grid_dims[1] * dx, self.grid_dims[2] * dx],
            device=positions.device,
            dtype=positions.dtype,
        )
        positions = torch.remainder(positions, domain)

        cv = float(cfg.c_v())
        total_internal = masses.to(torch.float32) * (cv * T_p.to(torch.float32))
        heats = torch.clamp(total_internal.to(self.dtype) - energies, min=0.0)

        return positions, velocities, energies, heats, excitations

    def _collide_spatial_hash(
        self,
        *,
        positions: "Tensor",
        velocities: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
        heats: "Tensor",
        cell_size: Optional[float] = None,
    ) -> None:
        raise RuntimeError(
            "_collide_spatial_hash() belonged to the legacy particle collision model. "
            "The single spatial model is now compressible ideal-gas Navier–Stokes + PIC."
        )

@dataclass
class SpectralCarrierConfig:
    """CUDA/Triton coherence field configuration (Ψ(ω) on a fixed ω lattice).

    Keep this aligned with the Metal config (same semantics, different backend).
    """

    max_carriers: int = 64
    coupling_scale: float = 0.25
    carrier_reg: float = 0.15
    temperature: float = 0.01

    conflict_threshold: float = 0.35
    offender_weight_floor: float = 1e-3
    ema_alpha: float = 0.10
    recenter_alpha: float = 0.10

    uncoupled_threshold: float = 0.1

    gate_width_init: float = 0.35
    gate_width_min: float = 0.08
    gate_width_max: float = 1.25

    # Memory + top-down bias (must match Metal semantics)
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

    # ------------------------------------------------------------------
    # Coherence field dynamics (dissipative GPE on a fixed ω lattice)
    # ------------------------------------------------------------------
    omega_min: Optional[float] = None
    omega_max: Optional[float] = None
    hbar_eff: float = 1.0
    mass_eff: float = 1.0
    g_interaction: float = -0.5
    energy_decay: float = 0.01
    chemical_potential: float = 0.0


class SpectralCarrierPhysics:
    """CUDA/Triton coherence field Ψ(ω) (complex ω-lattice dynamics)."""

    def __init__(
        self,
        config: SpectralCarrierConfig,
        grid_size: tuple[int, int, int],
        dt: float,
        device: str = "cuda",
        *,
        physical_constants: Optional[PhysicalConstants] = None,
        # Material parameters used to derive bath coupling from geometric physics.
        particle_radius: float = 0.5,
        dynamic_viscosity: float = 0.01,
        specific_heat: float = 10.0,
        grid_spacing: float = 1.0,
    ):
        if device != "cuda":
            raise RuntimeError(f"SpectralCarrierPhysics(CUDA) requires device='cuda', got '{device}'")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if int(config.anchor_slots) != k.ANCHORS:
            raise ValueError(
                f"SpectralCarrierConfig.anchor_slots must be {k.ANCHORS} "
                f"(got {config.anchor_slots})"
            )

        self.config = config
        self.grid_size = grid_size
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = torch.float32

        # Physics (simulation units). If not provided, fall back to SI identity mapping.
        self._const = physical_constants or PhysicalConstants.from_codata_si(UnitSystem.si())
        assert_finite_constants(self._const)

        # Geometric material parameters (used to derive a physically grounded bath rate).
        self._particle_radius = float(particle_radius)
        self._dynamic_viscosity = float(dynamic_viscosity)
        self._specific_heat = float(specific_heat)
        self._grid_spacing = float(grid_spacing)
        gx, gy, gz = self.grid_size
        self._domain_x = float(gx) * self._grid_spacing
        self._domain_y = float(gy) * self._grid_spacing
        self._domain_z = float(gz) * self._grid_spacing

        self.max_carriers = int(config.max_carriers)
        if self.max_carriers <= 0:
            raise ValueError("SpectralCarrierConfig.max_carriers must be > 0")

        # Carrier state buffers
        self.carrier_real = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_imag = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_omega = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.carrier_gate_width = torch.full(
            (self.max_carriers,), float(config.gate_width_init), device=self.device, dtype=self.dtype
        )
        self.carrier_conflict = torch.zeros(self.max_carriers, device=self.device, dtype=self.dtype)
        self.spawned_from_osc = torch.full((self.max_carriers,), -1, device=self.device, dtype=torch.int32)

        self.carrier_state = torch.zeros(self.max_carriers, device=self.device, dtype=torch.int32)
        self.carrier_age = torch.zeros(self.max_carriers, device=self.device, dtype=torch.int32)

        anchors = int(config.anchor_slots)
        self.anchor_slots = anchors
        self.carrier_anchor_idx = torch.full(
            (self.max_carriers * anchors,), -1, device=self.device, dtype=torch.int32
        )
        self.carrier_anchor_phase = torch.zeros(self.max_carriers * anchors, device=self.device, dtype=self.dtype)
        self.carrier_anchor_weight = torch.zeros(self.max_carriers * anchors, device=self.device, dtype=self.dtype)

        self._num_carriers_buf = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.num_carriers = 0

        self._random_phases = torch.rand(self.max_carriers, device=self.device, dtype=self.dtype)
        self._energy_stats = torch.zeros(4, device=self.device, dtype=self.dtype)
        self._rng_seed: int = 1

    def _ensure_seeded(self, osc_phase: "Tensor", osc_omega: "Tensor", osc_amp: "Tensor") -> None:
        if self.num_carriers > 0:
            return
        N = int(osc_phase.shape[0])
        if N == 0:
            return

        cfg = self.config

        idx0 = 0
        phi0 = float(osc_phase[idx0].item())
        omega0 = float(osc_omega[idx0].item())
        amp0 = float(osc_amp[idx0].item())

        if cfg.omega_min is not None and cfg.omega_max is not None:
            omega_min = float(cfg.omega_min)
            omega_max = float(cfg.omega_max)
        else:
            omega_min = float(osc_omega.detach().min().item())
            omega_max = float(osc_omega.detach().max().item())
            if not (omega_max > omega_min):
                omega_min -= 1.0
                omega_max += 1.0

        self.carrier_omega.copy_(
            torch.linspace(
                omega_min,
                omega_max,
                steps=int(self.max_carriers),
                device=self.device,
                dtype=self.dtype,
            )
        )

        sigma = float(cfg.gate_width_init)
        if sigma <= 0.0:
            sigma = max(1e-3, (omega_max - omega_min) / max(float(self.max_carriers), 1.0))
        env = torch.exp(-0.5 * ((self.carrier_omega - float(omega0)) / sigma) ** 2)
        self.carrier_real.copy_(env * (amp0 * math.cos(phi0)))
        self.carrier_imag.copy_(env * (amp0 * math.sin(phi0)))

        self.carrier_gate_width.fill_(float(cfg.gate_width_init))
        self.carrier_conflict.zero_()
        self.spawned_from_osc.fill_(-1)
        self.carrier_state.zero_()
        self.carrier_age.zero_()

        self.carrier_anchor_idx.fill_(-1)
        self.carrier_anchor_phase.zero_()
        self.carrier_anchor_weight.zero_()
        self.carrier_anchor_idx[0 :: self.anchor_slots] = idx0
        self.carrier_anchor_weight[0 :: self.anchor_slots] = env * float(amp0)

        self.num_carriers = int(self.max_carriers)
        self._num_carriers_buf[0] = int(self.max_carriers)

        if int(self.max_carriers) >= 2:
            domega = float((self.carrier_omega[1] - self.carrier_omega[0]).detach().item())
            self._inv_domega2 = (1.0 / (domega * domega)) if domega != 0.0 else 0.0
        else:
            self._inv_domega2 = 0.0

    def _set_energy_stats(self, energy: "Tensor") -> None:
        # [mean_abs, mean, std, count]
        eps = 1e-12
        mean_abs = energy.abs().mean()
        mean = energy.mean()
        var = (energy - mean).pow(2).mean()
        std = torch.sqrt(torch.clamp(var, min=0.0))
        count = torch.tensor(float(energy.numel()), device=energy.device, dtype=energy.dtype).clamp(min=1.0)
        self._energy_stats[0] = mean_abs
        self._energy_stats[1] = mean
        self._energy_stats[2] = std
        self._energy_stats[3] = count + eps  # keep nonzero

    def _params(
        self,
        *,
        mode: int,
        temperature: float,
        anchor_eps: float,
        offender_floor: float,
        rand_energy_eps: float,
        repulsion_scale: float,
    ) -> k.CoherenceParams:
        cfg = self.config
        return k.SpectralParams(
            dt=float(self.dt),
            coupling_scale=float(cfg.coupling_scale),
            carrier_reg=float(cfg.carrier_reg),
            temperature=float(temperature),
            rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
            conflict_threshold=float(cfg.conflict_threshold),
            offender_weight_floor=float(offender_floor),
            gate_width_min=float(cfg.gate_width_min),
            gate_width_max=float(cfg.gate_width_max),
            ema_alpha=float(cfg.ema_alpha),
            recenter_alpha=float(cfg.recenter_alpha),
            mode=int(mode),
            anchor_random_eps=float(anchor_eps),
            stable_amp_threshold=float(cfg.stable_amp_threshold),
            crystallize_amp_threshold=float(cfg.crystallize_amp_threshold),
            crystallize_conflict_threshold=float(cfg.crystallize_conflict_threshold),
            crystallize_age=int(cfg.crystallize_age),
            crystallized_coupling_boost=float(cfg.crystallized_coupling_boost),
            volatile_decay_mul=float(cfg.volatile_decay_mul),
            stable_decay_mul=float(cfg.stable_decay_mul),
            crystallized_decay_mul=float(cfg.crystallized_decay_mul),
            topdown_phase_scale=float(cfg.topdown_phase_scale),
            topdown_energy_scale=float(cfg.topdown_energy_scale),
            topdown_random_energy_eps=float(rand_energy_eps),
            repulsion_scale=float(repulsion_scale),
        )

    def step(
        self,
        osc_phase: "Tensor",
        particle_excitations: "Tensor",
        particle_energies: "Tensor",
        *,
        particle_positions: Optional["Tensor"] = None,
        particle_heats: Optional["Tensor"] = None,
        particle_masses: Optional["Tensor"] = None,
    ) -> Dict[str, "Tensor"]:
        osc_phase = osc_phase.to(device=self.device, dtype=self.dtype).contiguous()
        osc_omega = particle_excitations.to(device=self.device, dtype=self.dtype).contiguous()
        energy = particle_energies.to(device=self.device, dtype=self.dtype).contiguous()
        osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

        self._set_energy_stats(energy)
        self._ensure_seeded(osc_phase, osc_omega, osc_amp)
        if self.num_carriers == 0:
            return {
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

        # Advance RNG seed deterministically
        self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
        self._num_carriers_buf[0] = int(self.num_carriers)
        self._random_phases.uniform_()

        if particle_positions is None:
            raise ValueError(
                "SpectralCarrierPhysics.step requires particle_positions (N,3) to derive "
                "Hamiltonian coupling from overlap integrals."
            )
        pos = particle_positions.to(device=self.device, dtype=self.dtype).contiguous()

        if (
            particle_heats is None
            or particle_masses is None
            or particle_heats.numel() != energy.numel()
            or particle_masses.numel() != energy.numel()
        ):
            raise ValueError(
                "SpectralCarrierPhysics.step requires particle_heats and particle_masses "
                "(same shape as energies) to derive σ_x and bath rates."
            )

        const = self._const
        q = particle_heats.to(device=self.device, dtype=self.dtype).contiguous()
        m = particle_masses.to(device=self.device, dtype=self.dtype).contiguous()
        T_i = (q + energy) / (m * float(self._specific_heat))
        T_i = torch.clamp(T_i, min=0.0)
        T_bar = float(T_i.mean().detach().item()) if T_i.numel() else 0.0
        m_bar = float(m.mean().detach().item()) if m.numel() else 1.0

        spatial_sigma: float = 0.0
        cfg = self.config
        coupling_scale = float(cfg.coupling_scale)
        carrier_reg = float(cfg.carrier_reg)
        gate_width_init = float(cfg.gate_width_init)
        gate_width_min = float(cfg.gate_width_min)
        gate_width_max = float(cfg.gate_width_max)
        stable_amp_threshold = float(cfg.stable_amp_threshold)
        crystallize_amp_threshold = float(cfg.crystallize_amp_threshold)
        crystallize_age = int(cfg.crystallize_age)
        volatile_decay_mul = float(cfg.volatile_decay_mul)
        stable_decay_mul = float(cfg.stable_decay_mul)
        crystallized_decay_mul = float(cfg.crystallized_decay_mul)

        if T_bar > 0.0 and m_bar > 0.0 and float(const.hbar) > 0.0 and float(const.k_B) > 0.0:
            omega_th = (float(const.k_B) * T_bar) / float(const.hbar)
            tau_coh = float(const.hbar) / (float(const.k_B) * T_bar)
            bw_min = max(0.5 / float(self.dt), 1e-8)
            bw_max = max(float(omega_th), bw_min)

            spatial_sigma = (math.sqrt(2.0 * math.pi) * float(const.hbar)) / math.sqrt(
                max(m_bar * float(const.k_B) * T_bar, 1e-30)
            )

            mu = float(self._dynamic_viscosity) * math.sqrt(max(T_bar, 0.0))
            gamma = 6.0 * math.pi * mu * float(self._particle_radius)
            drag_rate = gamma / m_bar
            decoh_rate = max(float(drag_rate), (1.0 / tau_coh) if tau_coh > 0.0 else 0.0)
            decay = math.exp(-decoh_rate * float(self.dt)) if decoh_rate * float(self.dt) < 700 else 0.0

            zeta3 = 1.202056903159594
            kBT = float(const.k_B) * T_bar
            stable_amp_threshold = math.sqrt(max(kBT, 0.0))
            crystallize_amp_threshold = math.sqrt(max(zeta3 * kBT, 0.0))
            crystallize_age = max(1, int(math.ceil(tau_coh / float(self.dt)))) if math.isfinite(tau_coh) else 1

            coupling_scale = float(omega_th)
            carrier_reg = float(decoh_rate)
            gate_width_init = float(omega_th)
            gate_width_min = float(bw_min)
            gate_width_max = float(bw_max)
            volatile_decay_mul = float(decay)
            stable_decay_mul = float(decay)
            crystallized_decay_mul = 1.0

        params = self._params(
            mode=0,
            temperature=float(self.config.temperature),
            anchor_eps=float(self.config.anchor_random_eps),
            offender_floor=float(self.config.offender_weight_floor),
            rand_energy_eps=float(self.config.topdown_random_energy_eps),
            repulsion_scale=float(self.config.repulsion_scale),
        )
        params = k.SpectralParams(
            dt=float(params.dt),
            coupling_scale=float(coupling_scale),
            carrier_reg=float(carrier_reg),
            temperature=float(params.temperature),
            rng_seed=int(params.rng_seed),
            conflict_threshold=float(params.conflict_threshold),
            offender_weight_floor=float(params.offender_weight_floor),
            gate_width_min=float(gate_width_min),
            gate_width_max=float(gate_width_max),
            ema_alpha=float(params.ema_alpha),
            recenter_alpha=float(params.recenter_alpha),
            mode=int(params.mode),
            anchor_random_eps=float(params.anchor_random_eps),
            stable_amp_threshold=float(stable_amp_threshold),
            crystallize_amp_threshold=float(crystallize_amp_threshold),
            crystallize_conflict_threshold=float(params.crystallize_conflict_threshold),
            crystallize_age=int(crystallize_age),
            crystallized_coupling_boost=float(params.crystallized_coupling_boost),
            volatile_decay_mul=float(volatile_decay_mul),
            stable_decay_mul=float(stable_decay_mul),
            crystallized_decay_mul=float(crystallized_decay_mul),
            topdown_phase_scale=float(params.topdown_phase_scale),
            topdown_energy_scale=float(params.topdown_energy_scale),
            topdown_random_energy_eps=float(params.topdown_random_energy_eps),
            repulsion_scale=float(params.repulsion_scale),
        )

        # Single-model coherence: evolve a fixed ω-lattice Ψ(ω) (no conflict-driven splitting).
        k.coherence_gpe_step(
            osc_phase=osc_phase,
            osc_omega=osc_omega,
            osc_amp=osc_amp,
            particle_pos=pos,
            carrier_real=self.carrier_real,
            carrier_imag=self.carrier_imag,
            carrier_omega=self.carrier_omega,
            carrier_gate_width=self.carrier_gate_width,
            anchor_idx=self.carrier_anchor_idx,
            anchor_phase=self.carrier_anchor_phase,
            anchor_weight=self.carrier_anchor_weight,
            current_carriers=int(self.num_carriers),
            dt=float(self.dt),
            hbar_eff=float(self.config.hbar_eff),
            mass_eff=float(self.config.mass_eff),
            g_interaction=float(self.config.g_interaction),
            energy_decay=float(self.config.energy_decay),
            chemical_potential=float(self.config.chemical_potential),
            inv_domega2=float(getattr(self, "_inv_domega2", 0.0)),
            anchor_eps=float(self.config.anchor_random_eps),
            rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
            offender_weight_floor=float(self.config.offender_weight_floor),
            gate_width_min=float(gate_width_min),
            gate_width_max=float(gate_width_max),
            domain_x=float(self._domain_x),
            domain_y=float(self._domain_y),
            domain_z=float(self._domain_z),
            spatial_sigma=float(spatial_sigma),
        )

        # Top-down energy bias (crystallized carriers act as priors/completions)
        k.topdown_bias_energies(
            osc_energy=energy,
            osc_amp=osc_amp,
            carrier_state=self.carrier_state,
            anchor_idx=self.carrier_anchor_idx,
            anchor_weight=self.carrier_anchor_weight,
            num_carriers=self._num_carriers_buf,
            num_carriers_i=int(self.num_carriers),
            max_carriers=int(self.max_carriers),
            dt=float(self.dt),
            rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
            topdown_energy_scale=float(self.config.topdown_energy_scale),
            topdown_random_energy_eps=float(self.config.topdown_random_energy_eps),
        )

        osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

        # Oscillator phase update from carriers (torque + top-down phase pull)
        k.update_oscillator_phases(
            osc_phase=osc_phase,
            osc_omega=osc_omega,
            osc_amp=osc_amp,
            particle_pos=pos,
            carrier_real=self.carrier_real,
            carrier_imag=self.carrier_imag,
            carrier_omega=self.carrier_omega,
            carrier_gate_width=self.carrier_gate_width,
            carrier_state=self.carrier_state,
            anchor_idx=self.carrier_anchor_idx,
            anchor_phase=self.carrier_anchor_phase,
            anchor_weight=self.carrier_anchor_weight,
            energy_stats=self._energy_stats,
            num_carriers=self._num_carriers_buf,
            N=int(osc_phase.numel()),
            max_carriers=int(self.max_carriers),
            dt=float(self.dt),
            coupling_scale=float(self.config.coupling_scale),
            temperature=float(self.config.temperature),
            rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
            gate_width_min=float(self.config.gate_width_min),
            gate_width_max=float(self.config.gate_width_max),
            crystallized_coupling_boost=float(self.config.crystallized_coupling_boost),
            topdown_phase_scale=float(self.config.topdown_phase_scale),
            domain_x=float(self._domain_x),
            domain_y=float(self._domain_y),
            domain_z=float(self._domain_z),
            spatial_sigma=float(spatial_sigma),
        )

        # Single-model coherence: no carrier spawning/splitting. The ω-lattice is fixed.

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

    def idle_compute(
        self,
        osc_phase: "Tensor",
        particle_excitations: "Tensor",
        particle_energies: "Tensor",
        *,
        particle_positions: Optional["Tensor"] = None,
        particle_heats: Optional["Tensor"] = None,
        particle_masses: Optional["Tensor"] = None,
        steps: int = 1,
        mode: str = "explore",
    ) -> Dict[str, "Tensor"]:
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
            anchor_eps = max(float(self.config.anchor_random_eps), 0.20)
            rand_energy_eps = max(float(self.config.topdown_random_energy_eps), 0.10)
            offender_floor = float(self.config.offender_weight_floor) * 0.10
            repulsion = 0.0
        else:
            raise ValueError(f"Unknown idle_compute mode: {mode!r}")

        out: Dict[str, "Tensor"] = {}
        for _ in range(int(steps)):
            osc_phase = osc_phase.to(device=self.device, dtype=self.dtype).contiguous()
            osc_omega = particle_excitations.to(device=self.device, dtype=self.dtype).contiguous()
            energy = particle_energies.to(device=self.device, dtype=self.dtype).contiguous()
            osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

            self._set_energy_stats(energy)
            self._ensure_seeded(osc_phase, osc_omega, osc_amp)
            if self.num_carriers == 0:
                break

            self._rng_seed = (self._rng_seed + 1) & 0xFFFFFFFF
            self._num_carriers_buf[0] = int(self.num_carriers)
            self._random_phases.uniform_()

            if particle_positions is None:
                raise ValueError(
                    "SpectralCarrierPhysics.idle_compute requires particle_positions (N,3) to derive "
                    "Hamiltonian coupling from overlap integrals."
                )
            pos = particle_positions.to(device=self.device, dtype=self.dtype).contiguous()

            if (
                particle_heats is None
                or particle_masses is None
                or particle_heats.numel() != energy.numel()
                or particle_masses.numel() != energy.numel()
            ):
                raise ValueError(
                    "SpectralCarrierPhysics.idle_compute requires particle_heats and particle_masses "
                    "(same shape as energies) to derive σ_x and bath rates."
                )

            const = self._const
            q = particle_heats.to(device=self.device, dtype=self.dtype).contiguous()
            m_mass = particle_masses.to(device=self.device, dtype=self.dtype).contiguous()
            T_i = (q + energy) / (m_mass * float(self._specific_heat))
            T_i = torch.clamp(T_i, min=0.0)
            T_bar = float(T_i.mean().detach().item()) if T_i.numel() else 0.0
            m_bar = float(m_mass.mean().detach().item()) if m_mass.numel() else 1.0

            spatial_sigma: float = 0.0
            gate_width_init = float(self.config.gate_width_init)
            if T_bar > 0.0 and m_bar > 0.0 and float(const.hbar) > 0.0 and float(const.k_B) > 0.0:
                omega_th = (float(const.k_B) * T_bar) / float(const.hbar)
                gate_width_init = float(omega_th)
                spatial_sigma = (math.sqrt(2.0 * math.pi) * float(const.hbar)) / math.sqrt(
                    max(m_bar * float(const.k_B) * T_bar, 1e-30)
                )

            params = self._params(
                mode=m,
                temperature=temp,
                anchor_eps=anchor_eps,
                offender_floor=offender_floor,
                rand_energy_eps=rand_energy_eps,
                repulsion_scale=repulsion,
            )

            k.coherence_gpe_step(
                osc_phase=osc_phase,
                osc_omega=osc_omega,
                osc_amp=osc_amp,
                particle_pos=pos,
                carrier_real=self.carrier_real,
                carrier_imag=self.carrier_imag,
                carrier_omega=self.carrier_omega,
                carrier_gate_width=self.carrier_gate_width,
                anchor_idx=self.carrier_anchor_idx,
                anchor_phase=self.carrier_anchor_phase,
                anchor_weight=self.carrier_anchor_weight,
                current_carriers=int(self.num_carriers),
                dt=float(self.dt),
                hbar_eff=float(self.config.hbar_eff),
                mass_eff=float(self.config.mass_eff),
                g_interaction=float(self.config.g_interaction),
                energy_decay=float(self.config.energy_decay),
                chemical_potential=float(self.config.chemical_potential),
                inv_domega2=float(getattr(self, "_inv_domega2", 0.0)),
                anchor_eps=float(anchor_eps),
                rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
                offender_weight_floor=float(offender_floor),
                gate_width_min=float(self.config.gate_width_min),
                gate_width_max=float(self.config.gate_width_max),
                domain_x=float(self._domain_x),
                domain_y=float(self._domain_y),
                domain_z=float(self._domain_z),
                spatial_sigma=float(spatial_sigma),
            )

            k.topdown_bias_energies(
                osc_energy=energy,
                osc_amp=osc_amp,
                carrier_state=self.carrier_state,
                anchor_idx=self.carrier_anchor_idx,
                anchor_weight=self.carrier_anchor_weight,
                num_carriers=self._num_carriers_buf,
                num_carriers_i=int(self.num_carriers),
                max_carriers=int(self.max_carriers),
                dt=float(self.dt),
                rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
                topdown_energy_scale=float(self.config.topdown_energy_scale),
                topdown_random_energy_eps=float(rand_energy_eps),
            )

            osc_amp = torch.sqrt(torch.clamp(energy, min=1e-8)).contiguous()

            k.update_oscillator_phases(
                osc_phase=osc_phase,
                osc_omega=osc_omega,
                osc_amp=osc_amp,
                particle_pos=pos,
                carrier_real=self.carrier_real,
                carrier_imag=self.carrier_imag,
                carrier_omega=self.carrier_omega,
                carrier_gate_width=self.carrier_gate_width,
                carrier_state=self.carrier_state,
                anchor_idx=self.carrier_anchor_idx,
                anchor_phase=self.carrier_anchor_phase,
                anchor_weight=self.carrier_anchor_weight,
                energy_stats=self._energy_stats,
                num_carriers=self._num_carriers_buf,
                N=int(osc_phase.numel()),
                max_carriers=int(self.max_carriers),
                dt=float(self.dt),
                coupling_scale=float(self.config.coupling_scale),
                temperature=float(temp),
                rng_seed=int(self._rng_seed) & 0xFFFFFFFF,
                gate_width_min=float(self.config.gate_width_min),
                gate_width_max=float(self.config.gate_width_max),
                crystallized_coupling_boost=float(self.config.crystallized_coupling_boost),
                topdown_phase_scale=float(self.config.topdown_phase_scale),
                domain_x=float(self._domain_x),
                domain_y=float(self._domain_y),
                domain_z=float(self._domain_z),
                spatial_sigma=float(spatial_sigma),
            )

            # Single-model coherence: no carrier spawning/splitting. The ω-lattice is fixed.

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

    # -------------------------------------------------------------------------
    # Convergence predicates (wait-until conditions)
    # -------------------------------------------------------------------------

    def num_volatile(self) -> int:
        """Count of carriers in volatile state (state=0)."""
        if self.num_carriers == 0:
            return 0
        states = self.carrier_state[: self.num_carriers].detach().to("cpu")
        return int((states == 0).sum().item())

    def num_stable(self) -> int:
        """Count of carriers in stable state (state=1)."""
        if self.num_carriers == 0:
            return 0
        states = self.carrier_state[: self.num_carriers].detach().to("cpu")
        return int((states == 1).sum().item())

    def num_crystallized(self) -> int:
        """Count of carriers in crystallized state (state=2)."""
        if self.num_carriers == 0:
            return 0
        states = self.carrier_state[: self.num_carriers].detach().to("cpu")
        return int((states == 2).sum().item())

    def mean_conflict(self) -> float:
        """Mean conflict across active carriers (0 = fully coherent)."""
        if self.num_carriers == 0:
            return 0.0
        conf = self.carrier_conflict[: self.num_carriers].detach().to("cpu", dtype=torch.float32)
        return float(conf.mean().item())

    def max_conflict(self) -> float:
        """Maximum conflict across active carriers."""
        if self.num_carriers == 0:
            return 0.0
        conf = self.carrier_conflict[: self.num_carriers].detach().to("cpu", dtype=torch.float32)
        return float(conf.max().item())

    def mean_amplitude(self) -> float:
        """Mean amplitude across active carriers."""
        if self.num_carriers == 0:
            return 0.0
        cr = self.carrier_real[: self.num_carriers].detach().to("cpu", dtype=torch.float32)
        ci = self.carrier_imag[: self.num_carriers].detach().to("cpu", dtype=torch.float32)
        amp = torch.sqrt(cr * cr + ci * ci)
        return float(amp.mean().item())

    def carriers_stable(self, conflict_threshold: float = 0.15) -> bool:
        """True if all carriers have low conflict (phase coherence is good)."""
        if self.num_carriers == 0:
            return False
        return self.max_conflict() <= conflict_threshold

    def has_crystallized(self, min_count: int = 1) -> bool:
        """True if at least `min_count` carriers have crystallized."""
        return self.num_crystallized() >= min_count

    def all_stable_or_crystallized(self) -> bool:
        """True if no carriers are volatile (all have passed stable threshold)."""
        return self.num_volatile() == 0 and self.num_carriers > 0

    def thinking_complete(
        self,
        *,
        min_carriers: int = 1,
        max_conflict: float = 0.20,
        min_crystallized_frac: float = 0.0,
    ) -> bool:
        """Compound convergence check: carriers exist, conflict is low, optionally some crystallized."""
        if self.num_carriers < min_carriers:
            return False
        if self.mean_conflict() > max_conflict:
            return False
        if min_crystallized_frac > 0.0:
            frac = float(self.num_crystallized()) / float(self.num_carriers)
            if frac < min_crystallized_frac:
                return False
        return True

    def convergence_stats(self) -> dict:
        """Return a dictionary of convergence-related statistics."""
        return {
            "num_carriers": self.num_carriers,
            "num_volatile": self.num_volatile(),
            "num_stable": self.num_stable(),
            "num_crystallized": self.num_crystallized(),
            "mean_conflict": self.mean_conflict(),
            "max_conflict": self.max_conflict(),
            "mean_amplitude": self.mean_amplitude(),
        }


# Preferred names (describe the actual model).
CoherenceFieldConfig = SpectralCarrierConfig
CoherenceFieldPhysics = SpectralCarrierPhysics

# Preferred naming: hydrodynamic (ω-space) layer paired with ThermodynamicsDomain.
HydrodynamicDomainConfig = CoherenceFieldConfig
HydrodynamicDomain = CoherenceFieldPhysics

