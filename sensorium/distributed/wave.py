from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .particles import ParticleBatch
from .runtime import Transport


@dataclass(frozen=True)
class ShardedWaveConfig:
    total_modes: int
    omega_min: float
    omega_max: float
    dt_max: float = 0.01
    coupling: float = 0.1
    decay: float = 0.02


class ShardedWaveDomain:
    def __init__(
        self,
        *,
        config: ShardedWaveConfig,
        transport: Transport,
        device: torch.device,
    ) -> None:
        self.config = config
        self.transport = transport
        self.device = device
        self.dtype = torch.float32
        self.local_start, self.local_end = self._shard_bounds(
            rank=transport.rank,
            world_size=transport.world_size,
            total=config.total_modes,
        )
        self.local_modes = self.local_end - self.local_start
        self.omega_local = torch.linspace(
            config.omega_min,
            config.omega_max,
            config.total_modes,
            device=device,
            dtype=self.dtype,
        )[self.local_start : self.local_end].contiguous()
        self.psi_real = torch.zeros(
            (self.local_modes,), device=device, dtype=self.dtype
        )
        self.psi_imag = torch.zeros(
            (self.local_modes,), device=device, dtype=self.dtype
        )
        self.accum_real = torch.zeros(
            (self.local_modes,), device=device, dtype=self.dtype
        )
        self.accum_imag = torch.zeros(
            (self.local_modes,), device=device, dtype=self.dtype
        )

    def reset_accumulators(self) -> None:
        self.accum_real.zero_()
        self.accum_imag.zero_()

    def route_particle_mode_accumulations(
        self, tick: int, particles: ParticleBatch
    ) -> None:
        mode_idx = self._mode_index(particles.excitations)
        amp = torch.sqrt(torch.clamp(particles.energies, min=0.0) + 1e-8)
        contrib_real = amp * torch.cos(particles.phase)
        contrib_imag = amp * torch.sin(particles.phase)

        payloads_by_rank: dict[int, dict[str, torch.Tensor]] = {}
        for rank in range(self.transport.world_size):
            start, end = self._shard_bounds(
                rank=rank,
                world_size=self.transport.world_size,
                total=self.config.total_modes,
            )
            mask = (mode_idx >= start) & (mode_idx < end)
            if not torch.any(mask):
                continue
            payloads_by_rank[rank] = {
                "mode_idx": mode_idx[mask].to(torch.int64),
                "real": contrib_real[mask],
                "imag": contrib_imag[mask],
            }

        inbound = self.transport.route_mode_payloads(
            tick=tick,
            payloads_by_rank=payloads_by_rank,
            device=self.device,
        )
        for packet in inbound:
            if not packet:
                continue
            local_idx = (packet["mode_idx"] - self.local_start).to(torch.int64)
            self.accum_real.scatter_add_(0, local_idx, packet["real"])
            self.accum_imag.scatter_add_(0, local_idx, packet["imag"])

    def advance_wave(self, dt: float) -> None:
        dt_eff = min(float(dt), self.config.dt_max)
        omega = self.omega_local
        force_r = self.config.coupling * self.accum_real
        force_i = self.config.coupling * self.accum_imag
        dr = force_r - self.config.decay * self.psi_real - omega * self.psi_imag
        di = force_i - self.config.decay * self.psi_imag + omega * self.psi_real
        self.psi_real = self.psi_real + dt_eff * dr
        self.psi_imag = self.psi_imag + dt_eff * di

    def update_particle_phases(self, particles: ParticleBatch, dt: float) -> None:
        dt_eff = min(float(dt), self.config.dt_max)
        mode_idx = self._mode_index(particles.excitations)
        local_mask = (mode_idx >= self.local_start) & (mode_idx < self.local_end)
        if not torch.any(local_mask):
            particles.phase = particles.phase + particles.excitations * dt_eff
            return

        local_phase_push = torch.zeros_like(particles.phase)
        loc_idx = (mode_idx[local_mask] - self.local_start).to(torch.int64)
        psi_phase = torch.atan2(self.psi_imag[loc_idx], self.psi_real[loc_idx] + 1e-8)
        local_phase_push[local_mask] = psi_phase
        particles.phase = particles.phase + dt_eff * (
            particles.excitations + local_phase_push
        )

    def _mode_index(self, omega: torch.Tensor) -> torch.Tensor:
        span = max(self.config.omega_max - self.config.omega_min, 1e-6)
        norm = torch.clamp((omega - self.config.omega_min) / span, 0.0, 0.999999)
        idx = torch.floor(norm * float(self.config.total_modes)).to(torch.int64)
        return torch.clamp(idx, 0, self.config.total_modes - 1)

    @staticmethod
    def _shard_bounds(*, rank: int, world_size: int, total: int) -> tuple[int, int]:
        base = total // world_size
        rem = total % world_size
        start = rank * base + min(rank, rem)
        size = base + (1 if rank < rem else 0)
        end = start + size
        return int(start), int(end)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        amp = torch.sqrt(self.psi_real * self.psi_real + self.psi_imag * self.psi_imag)
        return {
            "psi_rms_local": torch.sqrt(torch.mean(amp * amp) + 1e-12),
            "psi_mean_local": torch.mean(amp),
            "mode_start": torch.tensor(float(self.local_start), device=self.device),
            "mode_end": torch.tensor(float(self.local_end), device=self.device),
        }
