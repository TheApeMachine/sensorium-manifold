from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..core.scatter import scatter_sum, segment_softmax


@dataclass
class PhysicsStepStats:
    edges: int
    heat_level: float
    sharpness: float
    energy_ratio: float


class ThermodynamicEngine:
    """Domain-agnostic thermodynamic engine with sparse interactions.

    Core design:
    - Interactions are expressed as an edge list (particle -> attractor).
    - The engine never materializes an NxM distance matrix unless a subclass chooses to.
    """

    def __init__(self, config: PhysicsConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.t = 0.0

        self.particles = BatchState.empty()
        self.attractors = BatchState.empty()

        # Homeostasis baseline (scalar)
        self._energy_baseline: Optional[torch.Tensor] = None

        # Slow-moving, global scales (EMA) to avoid per-step renormalization.
        self._dist_scale: Optional[torch.Tensor] = None
        self._pos_disp_scale: Optional[torch.Tensor] = None
        self._motion_scale: Optional[torch.Tensor] = None
        self._heat_abs_scale: Optional[torch.Tensor] = None
        self._energy_abs_scale: Optional[torch.Tensor] = None

        self.last_stats: Optional[PhysicsStepStats] = None

    def _ema_update(self, prev: Optional[torch.Tensor], cur: torch.Tensor, *, tau: float) -> torch.Tensor:
        dt = float(self.cfg.dt)
        if prev is None:
            return cur.detach().clone()
        alpha = dt / (float(tau) + dt)
        return (1.0 - alpha) * prev.to(cur.dtype) + alpha * cur.detach()

    def _update_global_scales(self, *, dists: Optional[torch.Tensor] = None, drift: Optional[torch.Tensor] = None) -> None:
        """Update slow global scales used inside inner loops.

        This decouples stabilizing normalization from per-step instantaneous means.
        """
        eps = self.cfg.eps
        med = self.cfg.medium
        tau = float(med.scale_tau)
        min_scale = float(med.min_scale)

        if dists is not None and dists.numel() > 0:
            cur = torch.std(dists.to(torch.float32)).clamp(min=min_scale) + eps
            self._dist_scale = self._ema_update(self._dist_scale, cur, tau=tau)

        # Position dispersion scale (for diffusion noise amplitude).
        if self.particles.has("position") and self.particles.get("position").numel() > 0:
            pos = self.particles.get("position").to(torch.float32)
            cur = torch.std(pos).clamp(min=min_scale) + eps
            self._pos_disp_scale = self._ema_update(self._pos_disp_scale, cur, tau=tau)

        # Motion scale (for motion->heat conversion).
        if drift is not None and drift.numel() > 0:
            if drift.ndim == 1:
                motion = drift.abs()
            else:
                motion = torch.linalg.norm(drift, dim=1)
            cur = motion.abs().mean().clamp(min=min_scale) + eps
            self._motion_scale = self._ema_update(self._motion_scale, cur, tau=tau)

        # Heat absolute scale (for cooling / diffusion normalization).
        heat_terms: list[torch.Tensor] = []
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            heat_terms.append(self.particles.get("heat").abs().mean().to(torch.float32))
        if self.attractors.has("heat") and self.attractors.get("heat").numel() > 0:
            heat_terms.append(self.attractors.get("heat").abs().mean().to(torch.float32))
        if heat_terms:
            cur = (torch.stack(heat_terms).mean()).clamp(min=min_scale) + eps
        else:
            cur = torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32).clamp(min=min_scale) + eps
        self._heat_abs_scale = self._ema_update(self._heat_abs_scale, cur, tau=tau)

        # Energy absolute scale (for energy damping normalization).
        energy_terms: list[torch.Tensor] = []
        if self.particles.has("energy") and self.particles.get("energy").numel() > 0:
            energy_terms.append(self.particles.get("energy").abs().mean().to(torch.float32))
        if self.attractors.has("energy") and self.attractors.get("energy").numel() > 0:
            energy_terms.append(self.attractors.get("energy").abs().mean().to(torch.float32))
        if energy_terms:
            cur_e = (torch.stack(energy_terms).mean()).clamp(min=min_scale) + eps
        else:
            cur_e = torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32).clamp(min=min_scale) + eps
        self._energy_abs_scale = self._ema_update(self._energy_abs_scale, cur_e, tau=tau)

    def _effective_homeostasis_tau(self, *, plasticity_gate: Optional[torch.Tensor]) -> float:
        """Return homeostasis baseline update time constant with optional plasticity."""
        tau = float(self.cfg.tau)
        if plasticity_gate is None:
            return tau
        g = float(plasticity_gate.detach().to(torch.float32).clamp(0.0, 1.0).item())
        return tau * (1.0 + float(self.cfg.homeostasis_tau_gain) * g)

    def _homeostasis_strength(self, *, plasticity_gate: Optional[torch.Tensor]) -> torch.Tensor:
        """Return multiplicative factor applied to damping strength (<= 1)."""
        if plasticity_gate is None:
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        g = plasticity_gate.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        return 1.0 / (1.0 + float(self.cfg.homeostasis_strength_gain) * g)

    # ----------------------------
    # Hooks for subclasses
    # ----------------------------

    def candidate_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (src_idx, dst_idx) edge list for particle-attractor interactions.

        Default: all-to-all (only appropriate for small systems).
        """
        n = self.particles.n
        m = self.attractors.n
        if n == 0 or m == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        src = torch.arange(n, device=self.device, dtype=torch.long).repeat_interleave(m)
        dst = torch.arange(m, device=self.device, dtype=torch.long).repeat(n)
        return src, dst

    def distance(self, p_pos: torch.Tensor, a_pos: torch.Tensor) -> torch.Tensor:
        """Per-edge distance metric. Subclasses should override."""
        d = p_pos - a_pos
        if d.ndim == 1:
            return d.abs()
        return torch.linalg.norm(d, dim=1)

    def post_step(self) -> None:
        """Optional cleanup after physics step (e.g., TTL)."""
        return

    # ----------------------------
    # Homeostasis
    # ----------------------------

    def total_energy(self) -> torch.Tensor:
        """Total energy proxy used for homeostasis."""
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("energy"):
            total = total + self.particles.get("energy").sum().to(torch.float32)
        if self.attractors.has("energy"):
            total = total + self.attractors.get("energy").sum().to(torch.float32)
        return total

    def _homeostasis_ratio(self, *, plasticity_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update baseline once and return E / baseline.

        Uses a fixed time constant to avoid runaway feedback loops.
        """
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        e = self.total_energy().to(torch.float32)
        if self._energy_baseline is None:
            self._energy_baseline = e.detach().clone()
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        base = self._energy_baseline.to(torch.float32)
        ratio = torch.log1p(e) / (torch.log1p(base) + eps)
        tau_eff = self._effective_homeostasis_tau(plasticity_gate=plasticity_gate)
        alpha = dt / (tau_eff + dt)
        self._energy_baseline = (1 - alpha) * base + alpha * e.detach()
        strength = self._homeostasis_strength(plasticity_gate=plasticity_gate)
        return (ratio * strength).clamp(min=eps)

    def step_physics(self) -> None:
        """Advance one thermodynamic step."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        med = self.cfg.medium

        n = self.particles.n
        m = self.attractors.n
        if n == 0 or m == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=1.0)
            return

        # Compute ratio once per step (thermostat).
        ratio = self._homeostasis_ratio().to(torch.float32)

        src, dst = self.candidate_edges()
        if src.numel() == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=float(ratio.item()))
            return

        p_pos = self.particles.get("position")[src]
        a_pos = self.attractors.get("position")[dst]

        dists = self.distance(p_pos, a_pos)  # [E]
        # Update slow-moving scales before using them.
        self._update_global_scales(dists=dists.to(torch.float32))

        d_scale = (self._dist_scale if self._dist_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
        sharpness = 1.0 / d_scale

        logits = -dists * sharpness
        w = segment_softmax(logits, src, n, eps=eps)  # [E], sum per particle = 1

        # Targets: weighted average of attractor positions per particle.
        a_pos_full = self.attractors.get("position")
        a_gather = a_pos_full[dst]
        if a_gather.ndim == 1:
            contrib = a_gather * w
        else:
            contrib = a_gather * w.unsqueeze(1)
        targets = scatter_sum(contrib, src, n)  # [N, ...]

        cur = self.particles.get("position")
        drift = targets - cur

        # Brownian diffusion scale emerges from current dispersion and heat.
        self._update_global_scales(drift=drift.to(torch.float32))
        disp = (self._pos_disp_scale if self._pos_disp_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
        noise = torch.randn_like(cur) * disp
        heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
        # Dimensionless temperature proxy: current heat relative to slow heat scale.
        heat_mean = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            heat_mean = self.particles.get("heat").abs().mean().to(torch.float32)
        elif self.attractors.has("heat") and self.attractors.get("heat").numel() > 0:
            heat_mean = self.attractors.get("heat").abs().mean().to(torch.float32)
        temperature = (heat_mean / heat_scale).clamp(min=0.0)
        noise = noise * (1.0 + temperature)

        visc = float(med.viscosity) if float(med.viscosity) > 0 else 1.0
        self.particles.set("position", cur + (dt / visc) * drift + dt * noise)

        # Particle heat update: motion -> heat; heat diffuses via binding.
        if self.particles.has("heat"):
            p_heat = self.particles.get("heat")
            motion = drift.abs() if drift.ndim == 1 else torch.linalg.norm(drift, dim=1)
            motion_scale = (self._motion_scale if self._motion_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
            p_heat = p_heat + dt * (motion / motion_scale)

            if self.attractors.has("heat"):
                a_heat = self.attractors.get("heat")
                # Project attractor heat onto particles via binding.
                h_in = scatter_sum((a_heat[dst] * w), src, n)
                heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
                p_heat = p_heat + (dt / float(med.thermal_resistance if float(med.thermal_resistance) > 0 else 1.0)) * (h_in - p_heat) / heat_scale

            # Homeostatic cooling.
            heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
            cool_tau = float(med.thermal_resistance if float(med.thermal_resistance) > 0 else 1.0) * heat_scale
            p_heat = p_heat * torch.exp(-dt * ratio.to(p_heat.dtype) / (cool_tau.to(p_heat.dtype) + eps))
            self.particles.set("heat", p_heat)

        # Attractor energy and heat receive weighted inflow.
        self.update_thermodynamics(src, dst, w, ratio=ratio.to(w.dtype))

        self.post_step()
        self.t += dt

        self.last_stats = PhysicsStepStats(
            edges=int(src.numel()),
            heat_level=float(temperature.detach().cpu().item()),
            sharpness=float(sharpness.detach().cpu().item()),
            energy_ratio=float(ratio.detach().cpu().item()),
        )

    def update_thermodynamics(self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Energy/heat flow between particles and attractors (edge-based)."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        med = self.cfg.medium

        m = self.attractors.n
        if m == 0:
            return

        # Energy inflow: weighted by particle energy if present; otherwise by binding mass.
        if self.particles.has("energy"):
            p_e = self.particles.get("energy")
            flow = w * p_e[src]
        else:
            flow = w

        e_in = scatter_sum(flow, dst, m)  # [M]

        a_e = self.attractors.ensure("energy", m, device=self.device, dtype=e_in.dtype)
        self._update_global_scales()
        e_scale = (self._energy_abs_scale if self._energy_abs_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
        a_e = (a_e + dt * e_in) * torch.exp(-dt * ratio / (e_scale.to(ratio.dtype) + eps))
        self.attractors.set("energy", a_e)

        if self.attractors.has("heat"):
            a_h = self.attractors.get("heat")
            if self.particles.has("heat"):
                p_h = self.particles.get("heat")
                h_flow = w * p_h[src]
            else:
                h_flow = w
            h_in = scatter_sum(h_flow, dst, m)
            h_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else torch.tensor(float(med.baseline_energy_scale), device=self.device, dtype=torch.float32)).clamp(min=float(med.min_scale)) + eps
            a_h = (a_h + dt * h_in) * torch.exp(-dt * ratio / (h_scale.to(ratio.dtype) + eps))
            self.attractors.set("heat", a_h)
