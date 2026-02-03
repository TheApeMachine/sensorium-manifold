from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..core.scatter import scatter_sum, segment_softmax


@dataclass
class PhysicsStepStats:
    """Statistics from a single physics step."""
    edges: int
    heat_level: float
    sharpness: float
    energy_ratio: float


class ThermodynamicEngine:
    """Domain-agnostic thermodynamic engine with sparse interactions.

    Core design:
    - Interactions are expressed as an edge list (particle -> attractor).
    - The engine never materializes an NxM distance matrix unless a subclass chooses to.
    
    Physics rules:
    1. Particles drift toward attractors (gravity-like)
    2. Motion generates heat
    3. Heat decreases viscosity (hotter = flows easier)
    4. Heat diffuses from hot to cold
    5. Homeostasis regulates energy toward baseline
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

    # ========================================
    # Scale Management (EMA tracking)
    # ========================================

    def _ema_update(self, prev: Optional[torch.Tensor], cur: torch.Tensor, *, tau: float) -> torch.Tensor:
        """Exponential moving average update."""
        dt = float(self.cfg.dt)
        if prev is None:
            return cur.detach().clone()
        alpha = dt / (float(tau) + dt)
        return (1.0 - alpha) * prev.to(cur.dtype) + alpha * cur.detach()

    def _get_scale(self, scale: Optional[torch.Tensor], default: float) -> torch.Tensor:
        """Get a scale tensor, falling back to default if not set."""
        if scale is not None:
            return scale.clamp(min=float(self.cfg.medium.min_scale)) + self.cfg.eps
        return torch.tensor(default, device=self.device, dtype=torch.float32).clamp(
            min=float(self.cfg.medium.min_scale)
        ) + self.cfg.eps

    def _update_distance_scale(self, dists: torch.Tensor) -> None:
        """Update distance scale from observed distances."""
        min_scale = float(self.cfg.medium.min_scale)
        tau = float(self.cfg.medium.scale_tau)
        eps = self.cfg.eps
        
        if dists.numel() > 1:
            cur = torch.std(dists.to(torch.float32)).clamp(min=min_scale) + eps
        elif dists.numel() == 1:
            cur = dists.abs().to(torch.float32).clamp(min=min_scale) + eps
        else:
            return
        self._dist_scale = self._ema_update(self._dist_scale, cur, tau=tau)

    def _update_position_scale(self) -> None:
        """Update position dispersion scale."""
        if not self.particles.has("position"):
            return
        pos = self.particles.get("position")
        if pos.numel() == 0:
            return
            
        min_scale = float(self.cfg.medium.min_scale)
        tau = float(self.cfg.medium.scale_tau)
        eps = self.cfg.eps
        
        if pos.numel() > 1:
            cur = torch.std(pos.to(torch.float32)).clamp(min=min_scale) + eps
        else:
            cur = torch.tensor(
                float(self.cfg.medium.baseline_energy_scale), 
                device=self.device, dtype=torch.float32
            ).clamp(min=min_scale) + eps
        self._pos_disp_scale = self._ema_update(self._pos_disp_scale, cur, tau=tau)

    def _update_motion_scale(self, drift: torch.Tensor) -> None:
        """Update motion scale from observed drift."""
        if drift.numel() == 0:
            return
            
        min_scale = float(self.cfg.medium.min_scale)
        tau = float(self.cfg.medium.scale_tau)
        eps = self.cfg.eps
        
        if drift.ndim == 1:
            motion = drift.abs()
        else:
            motion = torch.linalg.norm(drift, dim=1)
        cur = motion.abs().mean().clamp(min=min_scale) + eps
        self._motion_scale = self._ema_update(self._motion_scale, cur, tau=tau)

    def _update_heat_scale(self) -> None:
        """Update heat absolute scale."""
        min_scale = float(self.cfg.medium.min_scale)
        tau = float(self.cfg.medium.scale_tau)
        eps = self.cfg.eps
        
        heat_terms: list[torch.Tensor] = []
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            heat_terms.append(self.particles.get("heat").abs().mean().to(torch.float32))
        if self.attractors.has("heat") and self.attractors.get("heat").numel() > 0:
            heat_terms.append(self.attractors.get("heat").abs().mean().to(torch.float32))
        
        if heat_terms:
            cur = torch.stack(heat_terms).mean().clamp(min=min_scale) + eps
        else:
            cur = torch.tensor(
                float(self.cfg.medium.baseline_energy_scale), 
                device=self.device, dtype=torch.float32
            ).clamp(min=min_scale) + eps
        self._heat_abs_scale = self._ema_update(self._heat_abs_scale, cur, tau=tau)

    def _update_energy_scale(self) -> None:
        """Update energy absolute scale."""
        min_scale = float(self.cfg.medium.min_scale)
        tau = float(self.cfg.medium.scale_tau)
        eps = self.cfg.eps
        
        energy_terms: list[torch.Tensor] = []
        if self.particles.has("energy") and self.particles.get("energy").numel() > 0:
            energy_terms.append(self.particles.get("energy").abs().mean().to(torch.float32))
        if self.attractors.has("energy") and self.attractors.get("energy").numel() > 0:
            energy_terms.append(self.attractors.get("energy").abs().mean().to(torch.float32))
        
        if energy_terms:
            cur = torch.stack(energy_terms).mean().clamp(min=min_scale) + eps
        else:
            cur = torch.tensor(
                float(self.cfg.medium.baseline_energy_scale), 
                device=self.device, dtype=torch.float32
            ).clamp(min=min_scale) + eps
        self._energy_abs_scale = self._ema_update(self._energy_abs_scale, cur, tau=tau)

    # ========================================
    # Homeostasis
    # ========================================

    def total_energy(self) -> torch.Tensor:
        """Total energy proxy used for homeostasis."""
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("energy"):
            total = total + self.particles.get("energy").sum().to(torch.float32)
        if self.attractors.has("energy"):
            total = total + self.attractors.get("energy").sum().to(torch.float32)
        return total

    def _effective_homeostasis_tau(self, plasticity_gate: Optional[torch.Tensor]) -> float:
        """Return homeostasis baseline update time constant with optional plasticity."""
        tau = float(self.cfg.tau)
        if plasticity_gate is None:
            return tau
        g = float(plasticity_gate.detach().to(torch.float32).clamp(0.0, 1.0).item())
        return tau * (1.0 + float(self.cfg.homeostasis_tau_gain) * g)

    def _homeostasis_strength(self, plasticity_gate: Optional[torch.Tensor]) -> torch.Tensor:
        """Return multiplicative factor applied to damping strength (<= 1)."""
        if plasticity_gate is None:
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        g = plasticity_gate.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        return 1.0 / (1.0 + float(self.cfg.homeostasis_strength_gain) * g)

    def _homeostasis_ratio(self, plasticity_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute E / baseline ratio and update baseline via EMA."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        e = self.total_energy().to(torch.float32)
        
        if self._energy_baseline is None:
            self._energy_baseline = e.detach().clone()
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        
        base = self._energy_baseline.to(torch.float32)
        ratio = torch.log1p(e) / (torch.log1p(base) + eps)
        
        tau_eff = self._effective_homeostasis_tau(plasticity_gate)
        alpha = dt / (tau_eff + dt)
        self._energy_baseline = (1 - alpha) * base + alpha * e.detach()
        
        strength = self._homeostasis_strength(plasticity_gate)
        return (ratio * strength).clamp(min=eps)

    # ========================================
    # Temperature-Dependent Properties
    # ========================================

    def _compute_temperature(self) -> torch.Tensor:
        """Compute dimensionless temperature from heat."""
        heat_scale = self._get_scale(self._heat_abs_scale, self.cfg.medium.baseline_energy_scale)
        
        heat_mean = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.particles.has("heat") and self.particles.get("heat").numel() > 0:
            heat_mean = self.particles.get("heat").abs().mean().to(torch.float32)
        elif self.attractors.has("heat") and self.attractors.get("heat").numel() > 0:
            heat_mean = self.attractors.get("heat").abs().mean().to(torch.float32)
        
        return (heat_mean / heat_scale).clamp(min=0.0)

    def _effective_viscosity(self, temperature: torch.Tensor) -> torch.Tensor:
        """Heat decreases viscosity: hotter systems flow easier.
        
        Returns a tensor to keep computation on GPU.
        """
        base_visc = self.cfg.medium.viscosity if self.cfg.medium.viscosity > 0 else 1.0
        effective = base_visc / (1.0 + temperature)
        return effective.clamp(min=self.cfg.eps)

    def _effective_thermal_resistance(self, temperature: torch.Tensor) -> torch.Tensor:
        """Heat decreases thermal resistance: hotter systems conduct heat faster.
        
        Returns a tensor to keep computation on GPU.
        """
        base_res = self.cfg.medium.thermal_resistance if self.cfg.medium.thermal_resistance > 0 else 1.0
        effective = base_res / (1.0 + temperature)
        return effective.clamp(min=self.cfg.eps)

    # ========================================
    # Hooks for Subclasses
    # ========================================

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

    # ========================================
    # Core Physics Step
    # ========================================

    def step_physics(self) -> None:
        """Advance one thermodynamic step."""
        dt = float(self.cfg.dt)
        
        n = self.particles.n
        m = self.attractors.n
        
        # Early exit for empty system
        if n == 0 or m == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=1.0)
            return

        # 1. Compute homeostasis ratio (thermostat)
        ratio = self._homeostasis_ratio().to(torch.float32)

        # 2. Get candidate edges
        src, dst = self.candidate_edges()
        if src.numel() == 0:
            self.t += dt
            self.last_stats = PhysicsStepStats(edges=0, heat_level=0.0, sharpness=0.0, energy_ratio=float(ratio.item()))
            return

        # 3. Compute binding weights from distances
        dists, sharpness, w = self._compute_binding_weights(src, dst)

        # 4. Compute temperature
        temperature = self._compute_temperature()

        # 5. Update particle positions (drift + noise)
        self._update_particle_positions(src, dst, w, temperature)

        # 6. Update particle heat (motion -> heat, diffusion, cooling)
        self._update_particle_heat(src, dst, w, ratio, temperature)

        # 7. Update attractor energy and heat
        self._update_attractor_thermodynamics(src, dst, w, ratio)

        # 8. Post-step hook
        self.post_step()
        self.t += dt

        # 9. Record stats
        self.last_stats = PhysicsStepStats(
            edges=int(src.numel()),
            heat_level=float(temperature.detach().cpu().item()),
            sharpness=float(sharpness.detach().cpu().item()),
            energy_ratio=float(ratio.detach().cpu().item()),
        )

    def _compute_binding_weights(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute softmax binding weights from particle-attractor distances."""
        eps = self.cfg.eps
        n = self.particles.n
        
        p_pos = self.particles.get("position")[src]
        a_pos = self.attractors.get("position")[dst]
        dists = self.distance(p_pos, a_pos)
        
        # Update distance scale
        self._update_distance_scale(dists.to(torch.float32))
        
        d_scale = self._get_scale(self._dist_scale, self.cfg.medium.baseline_energy_scale)
        sharpness = 1.0 / d_scale
        
        logits = -dists * sharpness
        w = segment_softmax(logits, src, n, eps=eps)
        
        return dists, sharpness, w

    def _update_particle_positions(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, temperature: torch.Tensor
    ) -> None:
        """Update particle positions: drift toward attractors + Brownian noise."""
        dt = float(self.cfg.dt)
        n = self.particles.n
        
        # Compute weighted target positions
        a_pos_full = self.attractors.get("position")
        a_gather = a_pos_full[dst]
        if a_gather.ndim == 1:
            contrib = a_gather * w
        else:
            contrib = a_gather * w.unsqueeze(1)
        targets = scatter_sum(contrib, src, n)
        
        cur = self.particles.get("position")
        drift = targets - cur
        
        # Update scales
        self._update_motion_scale(drift.to(torch.float32))
        self._update_position_scale()
        
        # Brownian noise scaled by dispersion and temperature
        disp = self._get_scale(self._pos_disp_scale, self.cfg.medium.baseline_energy_scale)
        noise = torch.randn_like(cur) * disp * (1.0 + temperature)
        
        # Apply drift with temperature-dependent viscosity
        effective_visc = self._effective_viscosity(temperature)
        new_pos = cur + (dt / effective_visc) * drift + dt * noise
        
        self.particles.set("position", new_pos)

    def _update_particle_heat(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, 
        ratio: torch.Tensor, temperature: torch.Tensor
    ) -> None:
        """Update particle heat: motion generates heat, heat diffuses, homeostatic cooling."""
        if not self.particles.has("heat"):
            return
            
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        n = self.particles.n
        
        p_heat = self.particles.get("heat")
        
        # 1. Motion generates heat
        cur = self.particles.get("position")
        targets = self._compute_target_positions(src, dst, w)
        drift = targets - cur
        
        motion = drift.abs() if drift.ndim == 1 else torch.linalg.norm(drift, dim=1)
        motion_scale = self._get_scale(self._motion_scale, self.cfg.medium.baseline_energy_scale)
        p_heat = p_heat + dt * (motion / motion_scale)
        
        # 2. Heat diffuses from attractors to particles
        if self.attractors.has("heat"):
            a_heat = self.attractors.get("heat")
            h_in = scatter_sum(a_heat[dst] * w, src, n)
            heat_scale = self._get_scale(self._heat_abs_scale, self.cfg.medium.baseline_energy_scale)
            thermal_res = self._effective_thermal_resistance(temperature)
            p_heat = p_heat + (dt / thermal_res) * (h_in - p_heat) / heat_scale
        
        # 3. Homeostatic cooling
        heat_scale = self._get_scale(self._heat_abs_scale, self.cfg.medium.baseline_energy_scale)
        base_thermal_res = float(self.cfg.medium.thermal_resistance) if self.cfg.medium.thermal_resistance > 0 else 1.0
        cool_tau = base_thermal_res * heat_scale
        p_heat = p_heat * torch.exp(-dt * ratio.to(p_heat.dtype) / (cool_tau.to(p_heat.dtype) + eps))
        
        # Heat is always non-negative
        p_heat = p_heat.clamp(min=0.0)
        
        self.particles.set("heat", p_heat)

    def _compute_target_positions(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted average target positions for particles."""
        n = self.particles.n
        a_pos_full = self.attractors.get("position")
        a_gather = a_pos_full[dst]
        if a_gather.ndim == 1:
            contrib = a_gather * w
        else:
            contrib = a_gather * w.unsqueeze(1)
        return scatter_sum(contrib, src, n)

    def _update_attractor_thermodynamics(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, ratio: torch.Tensor
    ) -> None:
        """Update attractor energy and heat from particle inflow."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        m = self.attractors.n
        
        if m == 0:
            return
        
        # Update scales
        self._update_heat_scale()
        self._update_energy_scale()
        
        # Energy inflow
        self._update_attractor_energy(src, dst, w, ratio)
        
        # Heat inflow
        self._update_attractor_heat(src, dst, w, ratio)

    def _update_attractor_energy(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, ratio: torch.Tensor
    ) -> None:
        """Update attractor energy from particle inflow."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        m = self.attractors.n
        
        # Energy flow weighted by particle energy
        if self.particles.has("energy"):
            p_e = self.particles.get("energy")
            flow = w * p_e[src]
        else:
            flow = w
        
        e_in = scatter_sum(flow, dst, m)
        
        a_e = self.attractors.ensure("energy", m, device=self.device, dtype=e_in.dtype)
        e_scale = self._get_scale(self._energy_abs_scale, self.cfg.medium.baseline_energy_scale)
        
        # Add inflow and apply homeostatic damping
        a_e = (a_e + dt * e_in) * torch.exp(-dt * ratio / (e_scale.to(ratio.dtype) + eps))
        self.attractors.set("energy", a_e)

    def _update_attractor_heat(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, ratio: torch.Tensor
    ) -> None:
        """Update attractor heat from particle inflow."""
        if not self.attractors.has("heat"):
            return
            
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        m = self.attractors.n
        
        a_h = self.attractors.get("heat")
        
        # Heat flow weighted by particle heat
        if self.particles.has("heat"):
            p_h = self.particles.get("heat")
            h_flow = w * p_h[src]
        else:
            h_flow = w
        
        h_in = scatter_sum(h_flow, dst, m)
        h_scale = self._get_scale(self._heat_abs_scale, self.cfg.medium.baseline_energy_scale)
        
        # Add inflow and apply homeostatic damping
        a_h = (a_h + dt * h_in) * torch.exp(-dt * ratio / (h_scale.to(ratio.dtype) + eps))
        self.attractors.set("heat", a_h)

    # Legacy method for compatibility
    def update_thermodynamics(
        self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, *, ratio: torch.Tensor
    ) -> None:
        """Energy/heat flow between particles and attractors (edge-based).
        
        Deprecated: Use _update_attractor_thermodynamics instead.
        """
        self._update_attractor_thermodynamics(src, dst, w, ratio)
