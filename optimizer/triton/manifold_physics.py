"""CUDA/Triton implementation of manifold physics components.

This module mirrors the API shape of `optimizer/metal/manifold_physics.py` for CUDA.
Currently implemented:
- Spectral carriers (resonance potential) with crystallization + anchored top-down bias
- Idle compute modes (consolidate / disambiguate / explore)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import torch

from . import manifold_physics_kernels as k
from . import manifold_grid_kernels as g
from . import spatial_hash_kernels as sh

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ManifoldPhysicsConfig:
    """CUDA/Triton manifold physics simulation config (matches Metal semantics)."""

    grid_size: tuple[int, int, int] = (64, 64, 64)
    grid_spacing: float = 1.0
    dt: float = 0.01
    poisson_iterations: int = 50

    G: float = 0.001
    k_B: float = 0.1
    sigma_SB: float = 1e-5

    particle_radius: float = 0.5
    thermal_conductivity: float = 0.1
    specific_heat: float = 10.0
    dynamic_viscosity: float = 0.01
    emissivity: float = 0.5
    restitution: float = 0.8
    young_modulus: float = 1000.0


class ManifoldPhysics:
    """CUDA/Triton-accelerated manifold physics (grid + particles).

    Mirrors `optimizer/metal/manifold_physics.ManifoldPhysics`.
    """

    def __init__(self, config: ManifoldPhysicsConfig, device: str = "cuda"):
        if device != "cuda":
            raise RuntimeError(f"ManifoldPhysics(CUDA) requires device='cuda', got '{device}'")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32

        gx, gy, gz = config.grid_size
        self.grid_dims = (gx, gy, gz)

        self.gravity_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.gravity_potential = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)
        self.temperature_field = torch.zeros(gx, gy, gz, device=self.device, dtype=self.dtype)

    def scatter_particles(self, positions: "Tensor", masses: "Tensor", heats: "Tensor") -> None:
        g.clear_field(self.gravity_field.view(-1))
        g.clear_field(self.temperature_field.view(-1))
        g.scatter_particles(
            positions=positions.to(device=self.device, dtype=self.dtype),
            masses=masses.to(device=self.device, dtype=self.dtype),
            heats=heats.to(device=self.device, dtype=self.dtype),
            gravity_field=self.gravity_field,
            temperature_field=self.temperature_field,
            grid_spacing=float(self.config.grid_spacing),
        )

    def solve_gravity(self) -> None:
        cfg = self.config
        gravity_4pi = 4.0 * 3.14159265359 * float(cfg.G)
        phi_tmp = torch.zeros_like(self.gravity_potential)
        for i in range(int(cfg.poisson_iterations)):
            if i % 2 == 0:
                g.poisson_jacobi_step(
                    phi_in=self.gravity_potential,
                    rho=self.gravity_field,
                    phi_out=phi_tmp,
                    grid_spacing=float(cfg.grid_spacing),
                    gravity_4pi=float(gravity_4pi),
                )
            else:
                g.poisson_jacobi_step(
                    phi_in=phi_tmp,
                    rho=self.gravity_field,
                    phi_out=self.gravity_potential,
                    grid_spacing=float(cfg.grid_spacing),
                    gravity_4pi=float(gravity_4pi),
                )
        if int(cfg.poisson_iterations) % 2 == 1:
            self.gravity_potential.copy_(phi_tmp)

    def diffuse_heat(self) -> None:
        cfg = self.config
        temp_out = torch.zeros_like(self.temperature_field)
        g.diffuse_heat_field(
            temp_in=self.temperature_field,
            temp_out=temp_out,
            diffusion_coef=float(cfg.thermal_conductivity),
            dt=float(cfg.dt),
            grid_spacing=float(cfg.grid_spacing),
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
        cfg = self.config
        # in-place updates
        g.gather_update_particles(
            gravity_potential=self.gravity_potential,
            temperature_field=self.temperature_field,
            positions=positions.to(device=self.device, dtype=self.dtype),
            velocities=velocities.to(device=self.device, dtype=self.dtype),
            energies=energies.to(device=self.device, dtype=self.dtype),
            heats=heats.to(device=self.device, dtype=self.dtype),
            excitations=excitations.to(device=self.device, dtype=self.dtype),
            masses=masses.to(device=self.device, dtype=self.dtype),
            dt=float(cfg.dt),
            grid_spacing=float(cfg.grid_spacing),
            G=float(cfg.G),
            k_B=float(cfg.k_B),
            sigma_SB=float(cfg.sigma_SB),
            particle_radius=float(cfg.particle_radius),
            thermal_conductivity=float(cfg.thermal_conductivity),
            specific_heat=float(cfg.specific_heat),
            dynamic_viscosity=float(cfg.dynamic_viscosity),
            emissivity=float(cfg.emissivity),
            young_modulus=float(cfg.young_modulus),
        )
        return positions, velocities, energies, heats, excitations

    def step(
        self,
        positions: "Tensor",
        velocities: "Tensor",
        energies: "Tensor",
        heats: "Tensor",
        excitations: "Tensor",
        masses: "Tensor",
    ) -> tuple["Tensor", "Tensor", "Tensor", "Tensor", "Tensor"]:
        # 1) scatter
        self.scatter_particles(positions, masses, heats)
        # 2) solve fields
        self.solve_gravity()
        self.diffuse_heat()
        # 3) gather + update particles
        positions, velocities, energies, heats, excitations = self.gather_update_particles(
            positions, velocities, energies, heats, excitations, masses
        )
        # 4) collisions via spatial hash pipeline (CUDA)
        # Heuristic: enable spatial hash when N is moderately large.
        n = int(positions.shape[0])
        if n > 0:
            self._collide_spatial_hash(
                positions=positions,
                velocities=velocities,
                excitations=excitations,
                masses=masses,
                heats=heats,
            )
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
        cfg = self.config
        n = int(positions.shape[0])
        if n == 0:
            return
        if cell_size is None:
            cell_size = 4.0 * float(cfg.particle_radius)

        gx, gy, gz = self.grid_dims
        domain_size = (gx * float(cfg.grid_spacing), gy * float(cfg.grid_spacing), gz * float(cfg.grid_spacing))
        hash_x = max(1, int(math.ceil(domain_size[0] / cell_size)))
        hash_y = max(1, int(math.ceil(domain_size[1] / cell_size)))
        hash_z = max(1, int(math.ceil(domain_size[2] / cell_size)))
        num_cells = hash_x * hash_y * hash_z

        particle_cell_idx = torch.empty(n, device=self.device, dtype=torch.int32)
        cell_counts = torch.zeros(num_cells, device=self.device, dtype=torch.int32)

        sh.assign(
            positions=positions.to(device=self.device, dtype=self.dtype),
            particle_cell_idx=particle_cell_idx,
            cell_counts=cell_counts,
            hash_grid_x=hash_x,
            hash_grid_y=hash_y,
            hash_grid_z=hash_z,
            cell_size=float(cell_size),
            domain_min_x=0.0,
            domain_min_y=0.0,
            domain_min_z=0.0,
        )

        # Prefix sum (CUDA torch). cell_starts[i] = sum(counts[0:i])
        starts = torch.zeros(num_cells + 1, device=self.device, dtype=torch.int32)
        starts[1:] = torch.cumsum(cell_counts, dim=0)

        sorted_particle_idx = torch.empty(n, device=self.device, dtype=torch.int32)
        cell_offsets = starts[:num_cells].clone()
        sh.scatter(particle_cell_idx=particle_cell_idx, sorted_particle_idx=sorted_particle_idx, cell_offsets=cell_offsets)

        sh.collisions(
            positions=positions.to(device=self.device, dtype=self.dtype),
            velocities=velocities.to(device=self.device, dtype=self.dtype),
            excitations=excitations.to(device=self.device, dtype=self.dtype),
            masses=masses.to(device=self.device, dtype=self.dtype),
            heats=heats.to(device=self.device, dtype=self.dtype),
            sorted_particle_idx=sorted_particle_idx,
            cell_starts=starts,
            particle_cell_idx=particle_cell_idx,
            hash_grid_x=hash_x,
            hash_grid_y=hash_y,
            hash_grid_z=hash_z,
            cell_size=float(cell_size),
            domain_min_x=0.0,
            domain_min_y=0.0,
            domain_min_z=0.0,
            dt=float(cfg.dt),
            particle_radius=float(cfg.particle_radius),
            young_modulus=float(cfg.young_modulus),
            thermal_conductivity=float(cfg.thermal_conductivity),
            restitution=float(cfg.restitution),
            max_per_cell=64,
        )

@dataclass
class SpectralCarrierConfig:
    """CUDA/Triton spectral carrier configuration.

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


class SpectralCarrierPhysics:
    """CUDA/Triton spectral carriers (resonance potential, Langevin flow)."""

    def __init__(
        self,
        config: SpectralCarrierConfig,
        grid_size: tuple[int, int, int],
        dt: float,
        device: str = "cuda",
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
        idx = 0
        phi = float(osc_phase[idx].item())
        omega = float(osc_omega[idx].item())
        amp = float(osc_amp[idx].item())
        self.carrier_real[0] = amp * math.cos(phi)
        self.carrier_imag[0] = amp * math.sin(phi)
        self.carrier_omega[0] = omega
        self.carrier_gate_width[0] = float(self.config.gate_width_init)
        self.carrier_conflict[0] = 0.0
        self.spawned_from_osc[0] = 0
        self.carrier_state[0] = 0
        self.carrier_age[0] = 0

        self.carrier_anchor_idx.fill_(-1)
        self.carrier_anchor_phase.zero_()
        self.carrier_anchor_weight.zero_()
        self.carrier_anchor_idx[0] = 0
        self.carrier_anchor_phase[0] = 0.0
        self.carrier_anchor_weight[0] = amp

        self.num_carriers = 1
        self._num_carriers_buf[0] = 1

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
    ) -> k.SpectralParams:
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

        params = self._params(
            mode=0,
            temperature=float(self.config.temperature),
            anchor_eps=float(self.config.anchor_random_eps),
            offender_floor=float(self.config.offender_weight_floor),
            rand_energy_eps=float(self.config.topdown_random_energy_eps),
            repulsion_scale=float(self.config.repulsion_scale),
        )

        k.carrier_update_and_split(
            osc_phase=osc_phase,
            osc_omega=osc_omega,
            osc_amp=osc_amp,
            carrier_real=self.carrier_real,
            carrier_imag=self.carrier_imag,
            carrier_omega=self.carrier_omega,
            carrier_gate_width=self.carrier_gate_width,
            carrier_conflict=self.carrier_conflict,
            carrier_state=self.carrier_state,
            carrier_age=self.carrier_age,
            anchor_idx=self.carrier_anchor_idx,
            anchor_phase=self.carrier_anchor_phase,
            anchor_weight=self.carrier_anchor_weight,
            num_carriers=self._num_carriers_buf,
            spawned_from=self.spawned_from_osc,
            random_phases=self._random_phases,
            energy_stats=self._energy_stats,
            current_carriers=int(self.num_carriers),
            max_carriers=int(self.max_carriers),
            params=params,
        )

        new_count = int(self._num_carriers_buf.item())
        self.num_carriers = max(0, min(new_count, self.max_carriers))

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
        )

        # Spawn carriers for uncoupled oscillators
        k.spawn_uncoupled(
            osc_phase=osc_phase,
            osc_omega=osc_omega,
            osc_amp=osc_amp,
            carrier_real=self.carrier_real,
            carrier_imag=self.carrier_imag,
            carrier_omega=self.carrier_omega,
            carrier_gate_width=self.carrier_gate_width,
            carrier_conflict=self.carrier_conflict,
            carrier_state=self.carrier_state,
            carrier_age=self.carrier_age,
            anchor_idx=self.carrier_anchor_idx,
            anchor_phase=self.carrier_anchor_phase,
            anchor_weight=self.carrier_anchor_weight,
            num_carriers=self._num_carriers_buf,
            num_carriers_i=int(self.num_carriers),
            max_carriers=int(self.max_carriers),
            coupling_threshold=float(self.config.uncoupled_threshold),
            gate_width_init=float(self.config.gate_width_init),
            gate_width_min=float(self.config.gate_width_min),
            gate_width_max=float(self.config.gate_width_max),
        )
        self.num_carriers = int(self._num_carriers_buf.item())

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

            params = self._params(
                mode=m,
                temperature=temp,
                anchor_eps=anchor_eps,
                offender_floor=offender_floor,
                rand_energy_eps=rand_energy_eps,
                repulsion_scale=repulsion,
            )

            k.carrier_update_and_split(
                osc_phase=osc_phase,
                osc_omega=osc_omega,
                osc_amp=osc_amp,
                carrier_real=self.carrier_real,
                carrier_imag=self.carrier_imag,
                carrier_omega=self.carrier_omega,
                carrier_gate_width=self.carrier_gate_width,
                carrier_conflict=self.carrier_conflict,
                carrier_state=self.carrier_state,
                carrier_age=self.carrier_age,
                anchor_idx=self.carrier_anchor_idx,
                anchor_phase=self.carrier_anchor_phase,
                anchor_weight=self.carrier_anchor_weight,
                num_carriers=self._num_carriers_buf,
                spawned_from=self.spawned_from_osc,
                random_phases=self._random_phases,
                energy_stats=self._energy_stats,
                current_carriers=int(self.num_carriers),
                max_carriers=int(self.max_carriers),
                params=params,
            )

            new_count = int(self._num_carriers_buf.item())
            self.num_carriers = max(0, min(new_count, self.max_carriers))

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
            )

            k.spawn_uncoupled(
                osc_phase=osc_phase,
                osc_omega=osc_omega,
                osc_amp=osc_amp,
                carrier_real=self.carrier_real,
                carrier_imag=self.carrier_imag,
                carrier_omega=self.carrier_omega,
                carrier_gate_width=self.carrier_gate_width,
                carrier_conflict=self.carrier_conflict,
                carrier_state=self.carrier_state,
                carrier_age=self.carrier_age,
                anchor_idx=self.carrier_anchor_idx,
                anchor_phase=self.carrier_anchor_phase,
                anchor_weight=self.carrier_anchor_weight,
                num_carriers=self._num_carriers_buf,
                num_carriers_i=int(self.num_carriers),
                max_carriers=int(self.max_carriers),
                coupling_threshold=float(self.config.uncoupled_threshold),
                gate_width_init=float(self.config.gate_width_init),
                gate_width_min=float(self.config.gate_width_min),
                gate_width_max=float(self.config.gate_width_max),
            )
            self.num_carriers = int(self._num_carriers_buf.item())

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

