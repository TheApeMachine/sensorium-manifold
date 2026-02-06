"""Loader injects spatial properties into tokenizer output."""

import torch
from typing import Iterator
from sensorium.kernels.runtime import get_device
from sensorium.tokenizer.prototype import Tokenizer


class Loader:
    """Adds positions and velocities to token batches."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        grid_size: tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self._tokenizer = tokenizer
        self._grid_size = grid_size
        self._device = torch.device(get_device())
        self._dtype = torch.float32

    def stream(self) -> Iterator[dict]:
        """Yield state dicts with spatial injection."""
        gx, gy, gz = self._grid_size
        
        for batch in self._tokenizer.stream():
            osc, part = batch["oscillator"], batch["particle"]
            n = len(osc["token_ids"])
            if n == 0:
                continue
            
            # -----------------------------------------------------------------
            # Spatial injection (initial conditions; not learned)
            # -----------------------------------------------------------------
            # IMPORTANT: the simulation kernel uses physical coordinates with:
            #   Δx = 1 / max(grid_dims),  domain_d = grid_d * Δx  (≈ 1.0 for cubic grids)
            # Initialize positions directly in this physical domain so PIC indexing
            # and periodic wrapping are consistent from step 0.
            dx = 1.0 / float(max(gx, gy, gz))
            domain = torch.tensor([gx * dx, gy * dx, gz * dx], device=self._device, dtype=self._dtype)
            pos = torch.rand(n, 3, device=self._device, dtype=self._dtype) * domain

            # Initialize velocities with a bounded isotropic distribution.
            #
            # Rationale: a Gaussian has rare large outliers that can trip CFL / RK2
            # admissibility at startup (especially with many particles). Using a
            # bounded distribution gives predictable initialization without adding
            # any per-run "tuning knobs".
            v_max = 0.05  # [sim units] << 1 cell/step when dt≈Δx
            vel = (torch.rand(n, 3, device=self._device, dtype=self._dtype) * 2.0 - 1.0) * float(v_max)
            m = part.get("masses", None)
            if isinstance(m, torch.Tensor) and m.numel() == n:
                m = m.to(self._device, self._dtype)
                p_tot = (m[:, None] * vel).sum(dim=0)
                m_tot = m.sum()
                if bool((m_tot > 0).detach().item()):
                    vel = vel - (p_tot / m_tot)[None, :]

            yield {
                # Spatial (injected)
                "positions": pos,
                "velocities": vel,
                # Particle (from tokenizer)
                # NOTE: the gas grid evolves internal (thermal) energy density. Starting
                # from exactly-zero internal energy is a pressureless-degenerate regime
                # that is numerically fragile (any tiny negative overshoot becomes
                # inadmissible). We therefore inject a small baseline thermal energy
                # per particle as an initial condition.
                **{
                    **{k: v.to(self._device, self._dtype) for k, v in part.items()},
                    "heats": part["heats"].to(self._device, self._dtype) + 0.1,
                },
                # Oscillator (from tokenizer) 
                # Canonical wave-layer keys
                "phase": osc["phase"].to(self._device, self._dtype),
                "omega": osc["omega"].to(self._device, self._dtype),
                "energy_osc": osc["energy"].to(self._device, self._dtype),
                # Token identity (for crystal content decoding)
                "token_ids": osc["token_ids"].to(self._device, torch.int64),
                "sequence_indices": osc["sequence_indices"].to(self._device, torch.int64),
            }
