from __future__ import annotations

from dataclasses import dataclass

import torch

from .particles import ParticleBatch, ParticleMigrator
from .runtime import DistributedWorker, RankConfig, Transport
from .thermodynamics import DistributedThermoConfig, DistributedThermodynamicsDomain
from .wave import ShardedWaveConfig, ShardedWaveDomain


@dataclass(frozen=True)
class DistributedStepConfig:
    thermo: DistributedThermoConfig
    wave: ShardedWaveConfig
    dt: float = 0.005


class DistributedManifoldPipeline:
    def __init__(
        self,
        *,
        rank_config: RankConfig,
        transport: Transport,
        config: DistributedStepConfig,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.worker = DistributedWorker(rank_config=rank_config, transport=transport)
        self.worker.validate_rank()
        self.config = config
        self.thermo = DistributedThermodynamicsDomain(
            config=config.thermo,
            rank_config=rank_config,
            transport=transport,
            device=self.device,
        )
        self.migrator = ParticleMigrator(
            rank_config=rank_config,
            grid_spacing=config.thermo.grid_spacing,
            device=self.device,
        )
        self.wave = ShardedWaveDomain(
            config=config.wave,
            transport=transport,
            device=self.device,
        )
        self.particles = ParticleBatch(
            positions=torch.empty((0, 3), device=self.device, dtype=torch.float32),
            velocities=torch.empty((0, 3), device=self.device, dtype=torch.float32),
            masses=torch.empty((0,), device=self.device, dtype=torch.float32),
            heats=torch.empty((0,), device=self.device, dtype=torch.float32),
            energies=torch.empty((0,), device=self.device, dtype=torch.float32),
            excitations=torch.empty((0,), device=self.device, dtype=torch.float32),
            phase=torch.empty((0,), device=self.device, dtype=torch.float32),
        )
        self.tick = 0
        self._last_mode_global_stats = torch.zeros(
            (3,), device=self.device, dtype=torch.float32
        )
        self._last_wave_global_stats = torch.zeros(
            (3,), device=self.device, dtype=torch.float32
        )

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        self.particles = ParticleBatch.from_state(state, device=self.device)

    def deposit_particles_to_grid(self) -> None:
        self.thermo.rho_field.zero_()
        self.thermo.mom_field.zero_()
        self.thermo.e_int_field.zero_()
        if self.particles.size() == 0:
            self.thermo.rho_field.add_(self.config.thermo.rho_min)
            return

        rc = self.worker.rank_config
        dx = self.config.thermo.grid_spacing
        origin = torch.tensor(rc.tile_origin, device=self.device, dtype=torch.float32)
        local = self.particles.positions - origin.unsqueeze(0) * dx
        gx, gy, gz = rc.local_grid_size
        ix = torch.clamp((local[:, 0] / dx).floor().to(torch.int64), 0, gx - 1)
        iy = torch.clamp((local[:, 1] / dx).floor().to(torch.int64), 0, gy - 1)
        iz = torch.clamp((local[:, 2] / dx).floor().to(torch.int64), 0, gz - 1)

        lin = (ix * gy + iy) * gz + iz
        flat_rho = self.thermo.rho_field.view(-1)
        flat_e = self.thermo.e_int_field.view(-1)
        flat_mom = self.thermo.mom_field.view(-1, 3)
        flat_rho.scatter_add_(0, lin, self.particles.masses)
        flat_e.scatter_add_(0, lin, self.particles.energies)
        flat_mom.scatter_add_(
            0,
            lin.unsqueeze(-1).expand(-1, 3),
            self.particles.velocities * self.particles.masses.unsqueeze(-1),
        )
        self.thermo.rho_field.clamp_min_(self.config.thermo.rho_min)

    def exchange_grid_halos(self) -> None:
        self.thermo.exchange_grid_halos(self.tick)

    def advance_grid_interior(self) -> None:
        self.thermo.solve_gravity(self.tick)
        self.thermo.advance_grid_interior(self.config.dt)

    def advance_particles(self) -> None:
        if self.particles.size() == 0:
            return
        rc = self.worker.rank_config
        origin = torch.tensor(rc.tile_origin, device=self.device, dtype=torch.float32)
        local = (
            self.particles.positions
            - origin.unsqueeze(0) * self.config.thermo.grid_spacing
        )
        vel, heat = self.thermo.sample_grid(local)
        self.particles = self.migrator.advance_particles(
            self.particles,
            dt=self.config.dt,
            grid_velocity=vel,
            grid_heat=heat,
        )

    def pack_migrate_or_exchange_ghost_particles(self) -> None:
        kept, outbound = self.migrator.split_outbound(self.particles)
        inbound = self.worker.transport.exchange_particle_payloads(
            tick=self.tick,
            payloads=outbound,
            neighbors=self.worker.rank_config.neighbor_ids,
            device=self.device,
        )
        self.particles = self.migrator.merge_inbound(kept, inbound)

    def particle_to_mode_accumulate(self) -> None:
        self.wave.reset_accumulators()
        if self.particles.size() == 0:
            return
        self.wave.route_particle_mode_accumulations(self.tick, self.particles)

    def allreduce_mode_accums(self) -> None:
        local = torch.stack(
            (
                torch.sum(self.wave.accum_real),
                torch.sum(self.wave.accum_imag),
                torch.tensor(float(self.wave.accum_real.numel()), device=self.device),
            )
        ).to(torch.float32)
        self._last_mode_global_stats = self.worker.transport.allreduce_tensor_sum(local)

    def advance_wave(self) -> None:
        self.wave.advance_wave(self.config.dt)
        if self.particles.size() > 0:
            self.wave.update_particle_phases(self.particles, self.config.dt)

    def allreduce_wave_diagnostics(self) -> None:
        amp2 = (
            self.wave.psi_real * self.wave.psi_real
            + self.wave.psi_imag * self.wave.psi_imag
        )
        local = torch.stack(
            (
                torch.sum(amp2),
                torch.sum(torch.sqrt(torch.clamp(amp2, min=0.0))),
                torch.tensor(float(amp2.numel()), device=self.device),
            )
        ).to(torch.float32)
        self._last_wave_global_stats = self.worker.transport.allreduce_tensor_sum(local)

    def step(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.load_state(state)
        self.deposit_particles_to_grid()
        self.exchange_grid_halos()
        self.advance_grid_interior()
        self.advance_particles()
        self.pack_migrate_or_exchange_ghost_particles()
        self.particle_to_mode_accumulate()
        self.allreduce_mode_accums()
        self.advance_wave()
        self.allreduce_wave_diagnostics()
        self.tick += 1
        return self.state_dict()

    def state_dict(self) -> dict[str, torch.Tensor]:
        out = self.particles.to_state()
        out.update(
            {
                "rho_field": self.thermo.rho_field,
                "mom_field": self.thermo.mom_field,
                "e_int_field": self.thermo.e_int_field,
                "gravity_potential": self.thermo.gravity_potential,
                "psi_real_local": self.wave.psi_real,
                "psi_imag_local": self.wave.psi_imag,
                "mode_accum_global_real_sum": self._last_mode_global_stats[0],
                "mode_accum_global_imag_sum": self._last_mode_global_stats[1],
                "mode_accum_global_count": self._last_mode_global_stats[2],
                "psi_amp2_global_sum": self._last_wave_global_stats[0],
                "psi_amp_global_sum": self._last_wave_global_stats[1],
                "psi_global_count": self._last_wave_global_stats[2],
            }
        )
        out.update(self.wave.diagnostics())
        return out
