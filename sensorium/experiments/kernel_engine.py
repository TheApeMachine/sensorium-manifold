"""Shared kernel experiment utilities (Metal/MPS).

This provides a small "experiment runtime" for the kernel stack:
- inject token IDs as particle bursts into 3D space
- evolve manifold physics via Metal kernels
- evolve spectral carriers via Metal kernels
- score candidate IDs (and invert to byte values by brute force)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from optimizer.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierPhysics,
    SpectralCarrierConfig,
)


@dataclass(frozen=True, slots=True)
class KernelEngineConfig:
    device: str = "mps"
    grid_size: tuple[int, int, int] = (32, 32, 32)
    dt: float = 0.02

    # Injection
    particles_per_token: int = 8
    injection_spread: float = 0.6
    energy_scale: float = 1.0

    # Carrier cadence
    carrier_every: int = 1

    # Universal tokenizer hash (must match sensorium/core/tokenizer.py)
    hash_vocab_size: int = 4096
    hash_prime: int = 31
    special_size: int = 5  # <pad>,<unk>,<bos>,<eos>,<mask>
    num_labels: int = 0

    # ID -> omega mapping
    omega_range: float = 2.0
    omega_mod: int = 2048


def hash_id(byte_val: int, pos: int, *, hash_vocab_size: int, hash_prime: int, special_size: int) -> int:
    return int((int(byte_val) * int(hash_prime) + int(pos)) % int(hash_vocab_size) + int(special_size))


def label_id(label: int, *, special_size: int, hash_vocab_size: int) -> int:
    return int(int(special_size) + int(hash_vocab_size) + int(label))


def omega_from_id(token_id: int, *, omega_range: float, omega_mod: int) -> float:
    return float((int(token_id) % int(omega_mod)) / float(omega_mod) * float(omega_range))


def center_from_id(token_id: int, grid_size: tuple[int, int, int]) -> torch.Tensor:
    gx, gy, gz = grid_size
    x = (int(token_id) * 73856093) % gx
    y = (int(token_id) * 19349663) % gy
    z = (int(token_id) * 83492791) % gz
    return torch.tensor(
        [
            1 + (x % max(1, gx - 2)),
            1 + (y % max(1, gy - 2)),
            1 + (z % max(1, gz - 2)),
        ],
        dtype=torch.float32,
    )


class KernelTokenEngine:
    def __init__(
        self,
        cfg: KernelEngineConfig,
        *,
        physics_cfg: Optional[ManifoldPhysicsConfig] = None,
        spectral_cfg: Optional[SpectralCarrierConfig] = None,
    ):
        if cfg.device != "mps":
            raise RuntimeError("KernelTokenEngine currently expects device='mps' (Metal).")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")

        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dtype = torch.float32
        self.two_pi = float(2.0 * torch.pi)

        mp_cfg = physics_cfg or ManifoldPhysicsConfig(
            grid_size=cfg.grid_size,
            dt=float(cfg.dt),
            poisson_iterations=25,
            device=cfg.device,
        )
        self.physics = ManifoldPhysics(mp_cfg, device=cfg.device)

        scfg = spectral_cfg or SpectralCarrierConfig(max_carriers=64)
        self.carriers = SpectralCarrierPhysics(config=scfg, grid_size=cfg.grid_size, dt=float(cfg.dt), device=cfg.device)

        self.reset()

    def reset(self) -> None:
        self.positions = torch.empty((0, 3), device=self.dev, dtype=self.dtype)
        self.velocities = torch.empty((0, 3), device=self.dev, dtype=self.dtype)
        self.energies = torch.empty((0,), device=self.dev, dtype=self.dtype)
        self.heats = torch.empty((0,), device=self.dev, dtype=self.dtype)
        self.excitations = torch.empty((0,), device=self.dev, dtype=self.dtype)
        self.masses = torch.empty((0,), device=self.dev, dtype=self.dtype)
        self.osc_phase = torch.empty((0,), device=self.dev, dtype=self.dtype)
        self._last_carrier_state = None

    def inject_id(self, token_id: int, *, particles: Optional[int] = None, energy_scale: Optional[float] = None) -> None:
        n = int(particles if particles is not None else self.cfg.particles_per_token)
        en_scale = float(energy_scale if energy_scale is not None else self.cfg.energy_scale)

        center = center_from_id(int(token_id), self.cfg.grid_size).to(device=self.dev, dtype=self.dtype)
        pos = center.view(1, 3) + torch.randn(n, 3, device=self.dev, dtype=self.dtype) * float(self.cfg.injection_spread)
        pos = pos.clamp(0.5, float(min(self.cfg.grid_size) - 1.5))
        vel = torch.randn(n, 3, device=self.dev, dtype=self.dtype) * 0.05

        om = float(omega_from_id(int(token_id), omega_range=self.cfg.omega_range, omega_mod=self.cfg.omega_mod))
        ex = torch.full((n,), om, device=self.dev, dtype=self.dtype) + torch.randn(n, device=self.dev, dtype=self.dtype) * 0.01

        en = torch.full((n,), en_scale, device=self.dev, dtype=self.dtype)
        ht = torch.zeros((n,), device=self.dev, dtype=self.dtype)
        ms = en.clone()
        ph = torch.rand(n, device=self.dev, dtype=self.dtype) * self.two_pi

        self.positions = torch.cat([self.positions, pos], dim=0)
        self.velocities = torch.cat([self.velocities, vel], dim=0)
        self.energies = torch.cat([self.energies, en], dim=0)
        self.heats = torch.cat([self.heats, ht], dim=0)
        self.excitations = torch.cat([self.excitations, ex], dim=0)
        self.masses = torch.cat([self.masses, ms], dim=0)
        self.osc_phase = torch.cat([self.osc_phase, ph], dim=0)

    def step(self, t: int) -> None:
        self.positions, self.velocities, self.energies, self.heats, self.excitations = self.physics.step(
            self.positions,
            self.velocities,
            self.energies,
            self.heats,
            self.excitations,
            self.masses,
        )

        if (t % int(self.cfg.carrier_every)) == 0 and int(self.osc_phase.numel()) > 0:
            self._last_carrier_state = self.carriers.step(self.osc_phase, self.excitations, self.energies)
            self.osc_phase = self._last_carrier_state["osc_phase"]

    def score_ids(self, token_ids: torch.Tensor, *, carrier_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score candidate IDs via current carrier spectrum.

        Returns non-negative scores (higher is better).
        """
        if self._last_carrier_state is None:
            return torch.zeros_like(token_ids, dtype=torch.float32, device=token_ids.device)

        om_c = self._last_carrier_state["frequencies"]  # (M,)
        sig = self._last_carrier_state["gate_widths"].clamp(min=1e-3)  # (M,)
        amp = self._last_carrier_state["amplitudes"].clamp(min=0.0)  # (M,)
        if int(om_c.numel()) == 0:
            return torch.zeros_like(token_ids, dtype=torch.float32, device=token_ids.device)

        if carrier_mask is not None:
            carrier_mask = carrier_mask.to(device=om_c.device)
            if carrier_mask.dtype != torch.bool:
                carrier_mask = carrier_mask.to(torch.bool)
            if int(carrier_mask.numel()) == int(om_c.numel()):
                om_c = om_c[carrier_mask]
                sig = sig[carrier_mask]
                amp = amp[carrier_mask]
            if int(om_c.numel()) == 0:
                return torch.zeros_like(token_ids, dtype=torch.float32, device=token_ids.device)

        om_q = (token_ids.to(torch.long) % int(self.cfg.omega_mod)).to(torch.float32) / float(self.cfg.omega_mod) * float(self.cfg.omega_range)  # (K,)
        # tuning: (M,K)
        diff = om_c.view(-1, 1) - om_q.view(1, -1)
        tuning = torch.exp(-(diff * diff) / (sig.view(-1, 1) * sig.view(-1, 1)))
        scores = (amp.view(-1, 1) * tuning).sum(dim=0)
        return scores.to(torch.float32)

    def predict_byte(self, pos: int, *, carrier_mask: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """Predict next byte at sequence position `pos`.

        Returns (argmax_byte, scores[256]).
        """
        cand_bytes = torch.arange(256, device=self.dev, dtype=torch.long)
        cand_ids = (cand_bytes * int(self.cfg.hash_prime) + int(pos)) % int(self.cfg.hash_vocab_size)
        cand_ids = cand_ids + int(self.cfg.special_size)
        scores = self.score_ids(cand_ids, carrier_mask=carrier_mask)
        b = int(torch.argmax(scores).item())
        return b, scores

