from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.scatter import scatter_sum, segment_softmax
from ..core.diagnostics import BridgeDiagnosticsLogger


@dataclass
class BridgeOutput:
    """Spectral readout from the bridge."""

    spec_logits: torch.Tensor
    spec_probs: torch.Tensor
    meta: Dict[str, Any]


@dataclass
class BridgeObserveOutput:
    """Diagnostics from an observe() call."""

    mismatch: torch.Tensor
    mismatch_mean: float
    mismatch_min: float
    mismatch_max: float
    sem_entropy: float
    spec_entropy: float
    ratio: float
    heat_mean: float
    energy_mean: float


@dataclass
class BindEdges:
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    mass_in: torch.Tensor
    entropy: torch.Tensor


class BridgeManifold:
    """Vector-to-vector transduction via emergent carriers.

    This replaces the old ID->ID bipartite graph with a carrier population that couples
    a semantic vector space (R^D) to a spectral coordinate space (R).

    Design goals:
    - No dense semantic->spectral lookup table.
    - No backprop.
    - No Python-side per-edge loops.
    - Generalization via continuous semantic vectors.
    """

    def __init__(
        self,
        *,
        sem_dim: int,
        spec_bins: torch.Tensor,
        dt: float,
        device: torch.device,
        num_carriers: Optional[int] = None,
        eps: float = 1e-8,
        tau: float = 1.0,
        sem_horizon: bool = True,
        spec_horizon: bool = True,
        horizon_k: Optional[int] = None,
        diagnostics: Optional[BridgeDiagnosticsLogger] = None,
    ):
        self.device = device
        self.dt = float(dt)
        self.eps = float(eps)
        self.tau = float(tau)
        self.sem_horizon = bool(sem_horizon)
        self.spec_horizon = bool(spec_horizon)
        self.horizon_k = None if horizon_k is None else int(horizon_k)
        self._diagnostics = diagnostics
        self._observe_step = 0

        self.sem_dim = int(sem_dim)

        bins = spec_bins.to(device=device, dtype=torch.float32).flatten()
        if bins.numel() == 0:
            raise ValueError("spec_bins must be non-empty")
        # Keep bins sorted to enable event-horizon neighbor search.
        self.spec_bins, self._bin_order = torch.sort(bins)
        self.num_bins = int(self.spec_bins.numel())

        if num_carriers is None:
            # Emergent-ish default: carriers scale sublinearly with output resolution.
            num_carriers = max(1, int(round(math.sqrt(self.num_bins))))
        self.num_carriers = int(num_carriers)

        # Carrier state.
        self.sem_pos = torch.randn(self.num_carriers, self.sem_dim, device=device, dtype=torch.float32)
        self.sem_pos = self.sem_pos / (self.sem_pos.norm(dim=1, keepdim=True) + self.eps)

        # Initialize spectral positions by sampling from the output bin range.
        lo = float(self.spec_bins.min().item())
        hi = float(self.spec_bins.max().item())
        if math.isclose(lo, hi):
            self.spec_pos = torch.full((self.num_carriers,), lo, device=device, dtype=torch.float32)
        else:
            u = torch.rand(self.num_carriers, device=device, dtype=torch.float32)
            self.spec_pos = lo + (hi - lo) * u

        self.energy = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)
        self.heat = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)

        # Homeostasis baseline.
        self._energy_baseline: Optional[torch.Tensor] = None
        # Projection axis for semantic event-horizon selection.
        self._sem_axis = torch.randn(self.sem_dim, device=device, dtype=torch.float32)
        self._sem_axis = self._sem_axis / (self._sem_axis.norm() + self.eps)

    # ----------------------------
    # Homeostasis
    # ----------------------------

    def _ratio(self, total: torch.Tensor) -> torch.Tensor:
        """Return total / baseline with an emergent baseline timescale."""
        dt = self.dt
        eps = self.eps
        total = total.to(torch.float32)
        if self._energy_baseline is None:
            self._energy_baseline = total.detach().clone()
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        base = self._energy_baseline
        alpha = dt / (self.tau + dt)
        base_new = base * (1.0 - alpha) + total.detach() * alpha
        self._energy_baseline = base_new
        ratio = torch.log1p(total) / (torch.log1p(base_new) + eps)
        return ratio.clamp(min=eps)

    # ----------------------------
    # Locality helpers
    # ----------------------------

    def _neighbor_window_1d(self, query: torch.Tensor, key: torch.Tensor, k: int) -> torch.Tensor:
        if query.numel() == 0 or key.numel() == 0 or k <= 0:
            return torch.empty((int(query.numel()), 0), device=self.device, dtype=torch.long)
        m = int(key.numel())
        k = min(int(k), m)
        key_sorted, order = torch.sort(key)
        pos = torch.searchsorted(key_sorted, query)
        half = k // 2
        start = (pos - half).clamp(min=0, max=max(0, m - k))
        offsets = torch.arange(k, device=self.device, dtype=torch.long).unsqueeze(0)
        win = start.to(torch.long).unsqueeze(1) + offsets
        return order.index_select(0, win.reshape(-1)).reshape(win.shape)

    def _semantic_neighbors(self, sem_pos: torch.Tensor, k: int) -> torch.Tensor:
        proj_carriers = self.sem_pos @ self._sem_axis
        proj_query = sem_pos @ self._sem_axis
        return self._neighbor_window_1d(proj_query, proj_carriers, k)

    # ----------------------------
    # Core coupling primitives
    # ----------------------------

    def _semantic_bind(self, sem_pos: torch.Tensor, sem_energy: torch.Tensor) -> BindEdges:
        """Return sparse binding edges from semantic particles to carriers."""
        eps = self.eps

        sem_pos = sem_pos.to(device=self.device, dtype=torch.float32)
        sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()
        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        n = int(sem_pos.shape[0])
        if sem_pos.numel() == 0 or self.num_carriers == 0:
            mass_in = torch.zeros(self.num_carriers, device=self.device, dtype=torch.float32)
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return BindEdges(src=empty, dst=empty, weight=torch.empty(0, device=self.device), mass_in=mass_in, entropy=torch.tensor(0.0, device=self.device))

        if self.sem_horizon:
            k = self.horizon_k if self.horizon_k is not None else max(1, int(round(math.sqrt(self.num_carriers))))
            idx = self._semantic_neighbors(sem_pos, k)
        else:
            idx = torch.arange(self.num_carriers, device=self.device, dtype=torch.long).view(1, -1).expand(n, -1)

        # Distances on candidate edges.
        cand = self.sem_pos.index_select(0, idx.reshape(-1)).reshape(idx.shape + (-1,))
        diff = sem_pos.unsqueeze(1) - cand
        dist2 = (diff * diff).sum(dim=2)
        d_scale = torch.sqrt(dist2.mean() + eps)
        sharpness = 1.0 / (d_scale + eps)
        logits = -dist2 * sharpness
        w = torch.softmax(logits, dim=1)
        entropy = -(w * torch.log(w + eps)).sum(dim=1).mean()

        src = torch.arange(n, device=self.device, dtype=torch.long).repeat_interleave(idx.shape[1])
        dst = idx.reshape(-1)
        weight = w.reshape(-1)
        mass_in = scatter_sum(weight * sem_energy[src], dst, self.num_carriers)
        return BindEdges(src=src, dst=dst, weight=weight, mass_in=mass_in, entropy=entropy)

    def _spectral_bind(self, spec_pos: torch.Tensor, spec_energy: torch.Tensor) -> BindEdges:
        """Return sparse binding edges from spectral particles to carriers."""
        eps = self.eps

        spec_pos = spec_pos.to(device=self.device, dtype=torch.float32).flatten()
        spec_energy = spec_energy.to(device=self.device, dtype=torch.float32).flatten()
        n = int(spec_pos.shape[0])
        if spec_pos.numel() == 0 or self.num_carriers == 0:
            mass_in = torch.zeros(self.num_carriers, device=self.device, dtype=torch.float32)
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return BindEdges(src=empty, dst=empty, weight=torch.empty(0, device=self.device), mass_in=mass_in, entropy=torch.tensor(0.0, device=self.device))

        if self.spec_horizon:
            k = self.horizon_k if self.horizon_k is not None else max(1, int(round(math.sqrt(self.num_carriers))))
            idx = self._neighbor_window_1d(spec_pos, self.spec_pos, k)
        else:
            idx = torch.arange(self.num_carriers, device=self.device, dtype=torch.long).view(1, -1).expand(n, -1)

        cand = self.spec_pos.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        dist = (spec_pos.unsqueeze(1) - cand).abs()
        d_scale = dist.mean() + eps
        sharpness = 1.0 / d_scale
        logits = -dist * sharpness
        w = torch.softmax(logits, dim=1)
        entropy = -(w * torch.log(w + eps)).sum(dim=1).mean()

        src = torch.arange(n, device=self.device, dtype=torch.long).repeat_interleave(idx.shape[1])
        dst = idx.reshape(-1)
        weight = w.reshape(-1)
        mass_in = scatter_sum(weight * spec_energy[src], dst, self.num_carriers)
        return BindEdges(src=src, dst=dst, weight=weight, mass_in=mass_in, entropy=entropy)

    # ----------------------------
    # Learning / adaptation (no backprop)
    # ----------------------------

    def observe(
        self,
        *,
        sem_pos: torch.Tensor,
        sem_energy: Optional[torch.Tensor] = None,
        spec_pos: torch.Tensor,
        spec_energy: Optional[torch.Tensor] = None,
    ) -> BridgeObserveOutput:
        """Provide concurrent semantic + spectral evidence.

        This is the only "training" interface: the bridge updates its carriers
        based on co-activation, using purely local statistics.
        """

        eps = self.eps
        dt = self.dt

        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        if sem_energy is None:
            sem_energy = torch.ones(int(sem_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()

        spec_pos = spec_pos.to(device=self.device, dtype=torch.float32).flatten()
        if spec_energy is None:
            spec_energy = torch.ones(int(spec_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            spec_energy = spec_energy.to(device=self.device, dtype=torch.float32).flatten()

        # Normalize input energies so update magnitudes are scale-free.
        sem_energy = sem_energy.clamp(min=0.0)
        sem_energy = sem_energy / (sem_energy.sum() + eps)
        spec_energy = spec_energy.clamp(min=0.0)
        spec_energy = spec_energy / (spec_energy.sum() + eps)

        sem_bind = self._semantic_bind(sem_pos, sem_energy)
        spec_bind = self._spectral_bind(spec_pos, spec_energy)
        sem_mass = sem_bind.mass_in
        spec_mass = spec_bind.mass_in

        # Carrier targets in semantic space.
        if sem_bind.src.numel() > 0:
            sem_src = sem_pos[sem_bind.src]
            sem_w = (sem_energy[sem_bind.src] * sem_bind.weight).unsqueeze(1)
            sem_weighted = scatter_sum(sem_src * sem_w, sem_bind.dst, self.num_carriers)
            sem_target = sem_weighted / (sem_mass.unsqueeze(1) + eps)
        else:
            sem_target = self.sem_pos

        # Carrier targets in spectral space.
        if spec_bind.src.numel() > 0:
            spec_src = spec_pos[spec_bind.src]
            spec_w = spec_energy[spec_bind.src] * spec_bind.weight
            spec_weighted = scatter_sum(spec_src * spec_w, spec_bind.dst, self.num_carriers)
            spec_target = spec_weighted / (spec_mass + eps)
        else:
            spec_target = self.spec_pos

        # Co-activation signal (Hebbian, but at carrier level).
        sem_scale = sem_mass.mean() + eps
        spec_scale = spec_mass.mean() + eps
        sem_n = sem_mass / sem_scale
        spec_n = spec_mass / spec_scale
        co = sem_n * spec_n

        # Incoherence generates heat.
        mismatch = (sem_n - spec_n).abs()

        total = sem_energy.sum() + spec_energy.sum() + self.energy.abs().sum()
        ratio = self._ratio(total)

        # Position updates: relax toward current evidence.
        self.sem_pos = self.sem_pos + dt * (sem_target - self.sem_pos)
        self.sem_pos = self.sem_pos / (self.sem_pos.norm(dim=1, keepdim=True) + eps)

        self.spec_pos = self.spec_pos + dt * (spec_target - self.spec_pos)

        # Heat-driven diffusion (noise), without blurring the binding kernel.
        heat_level = self.heat.abs().mean()
        if bool(heat_level > 0):
            h_scale = heat_level / (heat_level + 1.0 + eps)
            self.sem_pos = self.sem_pos + dt * torch.randn_like(self.sem_pos) * h_scale
            self.spec_pos = self.spec_pos + dt * torch.randn_like(self.spec_pos) * h_scale

        # Energy + heat: homeostatic metabolism.
        e_scale = self.energy.abs().mean() + eps
        intake = co
        cost = ratio * self.energy / e_scale
        self.energy = (self.energy + dt * (intake - cost)).clamp(min=0.0)

        h_scale = self.heat.abs().mean() + eps
        self.heat = (self.heat + dt * (mismatch - ratio * self.heat / h_scale)).clamp(min=0.0)

        out = BridgeObserveOutput(
            mismatch=mismatch.detach(),
            mismatch_mean=float(mismatch.mean().item()),
            mismatch_min=float(mismatch.min().item()) if mismatch.numel() > 0 else 0.0,
            mismatch_max=float(mismatch.max().item()) if mismatch.numel() > 0 else 0.0,
            sem_entropy=float(sem_bind.entropy.item()),
            spec_entropy=float(spec_bind.entropy.item()),
            ratio=float(ratio.detach().item()),
            heat_mean=float(self.heat.mean().item()),
            energy_mean=float(self.energy.mean().item()),
        )
        if self._diagnostics is not None:
            self._diagnostics.log(step=self._observe_step, out=out)
        self._observe_step += 1
        return out

    def idle_think(self, steps: int = 1) -> None:
        """Self-observation loop to keep energy circulating without external input."""
        steps = int(steps)
        if steps <= 0 or self.num_carriers == 0:
            return
        eps = self.eps
        for _ in range(steps):
            sem_pos = self.sem_pos.detach()
            spec_pos = self.spec_pos.detach()
            sem_energy = self.energy.clamp(min=0.0)
            spec_energy = self.energy.clamp(min=0.0)
            h = self.heat.abs().mean()
            if bool(h > 0):
                noise_scale = h / (h + 1.0 + eps)
                sem_pos = sem_pos + noise_scale * torch.randn_like(sem_pos)
                spec_pos = spec_pos + noise_scale * torch.randn_like(spec_pos)
            self.observe(
                sem_pos=sem_pos,
                sem_energy=sem_energy,
                spec_pos=spec_pos,
                spec_energy=spec_energy,
            )

    def idle_think_from_semantic(self, sem_vec: torch.Tensor, steps: int = 1) -> None:
        """Idle bridge pondering driven by a semantic state vector (no new external data)."""
        steps = int(steps)
        if steps <= 0:
            return
        eps = self.eps
        sem_vec = sem_vec.to(device=self.device, dtype=torch.float32).view(1, -1)
        sem_energy = torch.ones(1, device=self.device, dtype=torch.float32)
        for _ in range(steps):
            # Use current bridge spectral state as concurrent evidence.
            spec_pos = self.spec_pos.detach()
            spec_energy = self.energy.clamp(min=0.0)
            if spec_energy.numel() > 0:
                spec_energy = spec_energy / (spec_energy.sum() + eps)
            self.observe(sem_pos=sem_vec, sem_energy=sem_energy, spec_pos=spec_pos, spec_energy=spec_energy)

    # ----------------------------
    # Readout
    # ----------------------------

    def forward(self, sem_pos: torch.Tensor, sem_energy: Optional[torch.Tensor] = None) -> BridgeOutput:
        """Project semantic particle(s) into a spectral distribution over spec_bins."""

        eps = self.eps

        if sem_pos.ndim == 1:
            sem_pos = sem_pos.view(1, -1)
        sem_pos = sem_pos.to(device=self.device, dtype=torch.float32)
        if sem_energy is None:
            sem_energy = torch.ones(int(sem_pos.shape[0]), device=self.device, dtype=torch.float32)
        else:
            sem_energy = sem_energy.to(device=self.device, dtype=torch.float32).flatten()
        sem_energy = sem_energy.clamp(min=0.0)
        sem_energy = sem_energy / (sem_energy.sum() + eps)

        # Carrier activations from semantic input.
        sem_bind = self._semantic_bind(sem_pos, sem_energy)
        carrier_act = sem_bind.mass_in * (self.energy / (self.energy.mean() + eps))
        carrier_act = carrier_act.clamp(min=0.0)

        if carrier_act.numel() == 0 or float(carrier_act.sum().item()) <= float(eps):
            logits = torch.zeros(self.num_bins, device=self.device, dtype=torch.float32)
            probs = logits
            meta = {"carriers": int(self.num_carriers), "active": 0}
            return BridgeOutput(spec_logits=logits, spec_probs=probs, meta=meta)

        # Event-horizon projection: each carrier distributes mass to its left/right bin neighbors.
        bins = self.spec_bins
        m = self.num_bins
        a_sorted = bins
        p = self.spec_pos
        ins = torch.searchsorted(a_sorted, p)
        left = (ins - 1).clamp(min=0, max=m - 1)
        right = ins.clamp(min=0, max=m - 1)
        src = torch.arange(self.num_carriers, device=self.device, dtype=torch.long)
        src2 = torch.cat([src, src], dim=0)
        dst2 = torch.cat([left, right], dim=0)

        # Remove duplicate edges where left==right.
        keep = torch.ones(src2.numel(), device=self.device, dtype=torch.bool)
        dup = left == right
        keep[self.num_carriers :][dup] = False
        src_e = src2[keep]
        dst_e = dst2[keep]

        # Distances on the spectral axis.
        d = (p[src_e] - a_sorted[dst_e]).abs()
        d_scale = d.mean() + eps
        sharpness = 1.0 / d_scale
        logits_e = -d * sharpness

        # Softmax within each carrier's outgoing neighborhood.
        w_e = segment_softmax(logits_e, src_e, self.num_carriers, eps=eps)

        # Scatter to bins.
        contrib = carrier_act[src_e] * w_e
        logits = scatter_sum(contrib, dst_e, m)
        probs = logits / (logits.sum() + eps)

        meta = {
            "carriers": int(self.num_carriers),
            "active": int(torch.count_nonzero(carrier_act).item()),
            "energy_mean": float(self.energy.mean().item()),
            "heat_mean": float(self.heat.mean().item()),
        }
        return BridgeOutput(spec_logits=logits, spec_probs=probs, meta=meta)
