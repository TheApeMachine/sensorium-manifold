from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..physics.engine import ThermodynamicEngine


@dataclass
class SpectralOutput:
    frequencies: torch.Tensor
    amplitudes: torch.Tensor
    meta: Dict[str, Any]


class SpectralManifold(ThermodynamicEngine):
    """1D thermodynamic diffusion over frequency attractors."""

    def __init__(self, config: PhysicsConfig, device: torch.device):
        super().__init__(config, device)

        self.particles = BatchState.empty()
        self.attractors = BatchState.empty()

    def set_targets(self, freqs: torch.Tensor, energy: Optional[torch.Tensor] = None) -> None:
        freqs = freqs.to(device=self.device, dtype=torch.float32).flatten()
        m = int(freqs.numel())
        if m == 0:
            self.attractors = BatchState.empty()
            return
        if energy is None:
            energy = torch.ones(m, device=self.device, dtype=torch.float32) / (m + self.cfg.eps)
        else:
            energy = energy.to(device=self.device, dtype=torch.float32).flatten()
            energy = energy / (energy.sum() + self.cfg.eps)

        self.attractors = BatchState(
            {
                "position": freqs.clamp(min=self.cfg.eps),
                "energy": energy,
                "heat": torch.zeros(m, device=self.device, dtype=torch.float32),
            }
        )

    def seed_particles(self, n: int) -> None:
        n = int(n)
        if n <= 0 or self.attractors.n == 0:
            self.particles = BatchState.empty()
            return
        freqs = self.attractors.get("position")
        probs = self.attractors.get("energy").clamp(min=0.0)
        probs = probs / (probs.sum() + self.cfg.eps)
        idx = torch.multinomial(probs, n, replacement=True)
        base = freqs[idx]

        # Dispersion emerges from target spread.
        spread = freqs.std() + self.cfg.eps
        noise = torch.randn(n, device=self.device) * spread
        pos = (base + noise).clamp(min=self.cfg.eps)

        self.particles = BatchState(
            {
                "position": pos,
                "energy": torch.ones(n, device=self.device, dtype=torch.float32) / (n + self.cfg.eps),
                "heat": torch.zeros(n, device=self.device, dtype=torch.float32),
            }
        )

    # ----------------------------
    # Engine overrides
    # ----------------------------

    def candidate_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.particles.n
        m = self.attractors.n
        if n == 0 or m == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        a = self.attractors.get("position")
        a_sorted, order = torch.sort(a)
        p = self.particles.get("position")
        # insertion indices in sorted attractors
        ins = torch.searchsorted(a_sorted, p)

        left = (ins - 1).clamp(min=0, max=m - 1)
        right = ins.clamp(min=0, max=m - 1)

        # Map back to original attractor indices.
        left_idx = order[left]
        right_idx = order[right]

        src = torch.arange(n, device=self.device, dtype=torch.long)
        src2 = torch.cat([src, src], dim=0)
        dst2 = torch.cat([left_idx, right_idx], dim=0)

        # Remove duplicates (when left==right).
        keep = torch.ones(src2.numel(), device=self.device, dtype=torch.bool)
        # duplicates occur at positions i and i+n
        dup = left_idx == right_idx
        keep[n:][dup] = False

        return src2[keep], dst2[keep]

    def distance(self, p_pos: torch.Tensor, a_pos: torch.Tensor) -> torch.Tensor:
        return (p_pos - a_pos).abs()

    def post_step(self) -> None:
        # Ensure positive frequencies.
        if self.particles.n:
            self.particles.set("position", self.particles.get("position").clamp(min=self.cfg.eps))

    # ----------------------------
    # Readout
    # ----------------------------

    def output_state(self, topk: Optional[int] = None) -> SpectralOutput:
        if self.particles.n == 0:
            return SpectralOutput(
                frequencies=torch.empty(0, device=self.device),
                amplitudes=torch.empty(0, device=self.device),
                meta={"empty": True},
            )
        freqs = self.particles.get("position")
        amps = self.particles.get("energy")
        if topk is not None and freqs.numel() > int(topk):
            k = int(topk)
            idx = torch.topk(amps, k=k).indices
            freqs = freqs[idx]
            amps = amps[idx]
        meta = {
            "num_particles": int(self.particles.n),
            "num_targets": int(self.attractors.n),
        }
        return SpectralOutput(frequencies=freqs, amplitudes=amps, meta=meta)
