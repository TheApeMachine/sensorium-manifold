from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

import torch

from ..core.config import PhysicsConfig
from ..core.diagnostics import SemanticDiagnosticsLogger
from ..core.state import BatchState
from ..physics.engine import ThermodynamicEngine
from .bond_graph import SparseBondGraph
from .carriers import CarrierPool, ParticleCarrierBonds


@dataclass
class SemanticOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    token_index: int
    token: Optional[str]
    meta: Dict[str, Any]


class SemanticManifold(ThermodynamicEngine):
    """Thermodynamic grammar on a sparse bond graph (no dense V×V matrices)."""

    def __init__(self, config: PhysicsConfig, device: torch.device, *, vocab: List[str], embed_dim: Optional[int] = None):
        super().__init__(config, device)
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.embed_dim = int(embed_dim if embed_dim is not None else self.vocab_size)

        pos = self._init_embeddings(self.vocab_size, self.embed_dim, eps=self.cfg.eps, device=device)
        self.attractors = BatchState(
            {
                "id": torch.arange(self.vocab_size, device=device, dtype=torch.long),
                "position": pos,
                "energy": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
                "excitation": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
                "heat": torch.zeros(self.vocab_size, device=device, dtype=torch.float32),
            }
        )

        self.graph = SparseBondGraph(self.vocab_size, device=device, dtype=torch.float32, eps=self.cfg.eps)
        
        # Carrier-based energy transport
        # Number of carriers scales with sqrt of vocab size (emergent)
        num_carriers = max(16, int(self.vocab_size ** 0.5))
        self.carriers = CarrierPool(
            num_carriers=num_carriers,
            embed_dim=self.embed_dim,
            device=device,
            eps=self.cfg.eps,
        )
        self.particle_carrier_bonds = ParticleCarrierBonds(
            num_particles=self.vocab_size,
            num_carriers=num_carriers,
            device=device,
            eps=self.cfg.eps,
        )
        # How many carriers each particle bonds to
        self.bonds_per_particle = min(8, num_carriers)

        # Pondering state (idle discovery)
        self.hunger = torch.zeros(self.vocab_size, device=device, dtype=torch.float32)
        self._surprise_baseline: Optional[torch.Tensor] = None
        self._ponder_step = 0
        self._diagnostics: Optional[SemanticDiagnosticsLogger] = None
        self._fresh_observation = False

        # Thinking / halting state
        self.halt_mass = 0.0
        self.last_entropy: Optional[float] = None
        self.last_confidence: float = 0.0
        self.last_debug: Dict[str, float] = {}

        # Per-carrier (token) baselines for local homeostasis.
        self._excitation_baseline: Optional[torch.Tensor] = None
        self._heat_baseline: Optional[torch.Tensor] = None

        # Plasticity gate (e.g. surprise/mismatch) used to modulate homeostasis.
        self._plasticity_gate: Optional[torch.Tensor] = None

        # Idle pondering gate
        self._last_activity_time = time.time()
        
        # Energy conservation tracking
        self._total_energy_input = 0.0  # Cumulative energy from data
        self._conservation_violations: List[Dict[str, float]] = []
        self._conservation_tolerance = 1e-3  # Relative tolerance for conservation check

    def _mark_activity(self) -> None:
        self._last_activity_time = time.time()
    
    # ----------------------------
    # Conservation Validation
    # ----------------------------
    
    def validate_conservation(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Validate that energy + heat equals total energy input.
        
        Returns dict with:
        - valid: bool - whether conservation holds within tolerance
        - total_input: float - cumulative energy from data
        - current_energy: float - current energy in system
        - current_heat: float - current heat in system
        - current_total: float - energy + heat
        - error: float - absolute difference
        - relative_error: float - error / total_input
        """
        energy = self.attractors.get("energy")
        heat = self.attractors.get("heat")
        
        current_energy = float(energy.sum().item())
        current_heat = float(heat.sum().item())
        current_total = current_energy + current_heat
        
        error = abs(current_total - self._total_energy_input)
        relative_error = error / (self._total_energy_input + 1e-8)
        valid = relative_error <= self._conservation_tolerance
        
        result = {
            "valid": valid,
            "step": step,
            "total_input": self._total_energy_input,
            "current_energy": current_energy,
            "current_heat": current_heat,
            "current_total": current_total,
            "error": error,
            "relative_error": relative_error,
        }
        
        if not valid:
            self._conservation_violations.append(result)
        
        return result
    
    def get_conservation_report(self) -> Dict[str, Any]:
        """Get a summary report of energy conservation."""
        current = self.validate_conservation()
        return {
            "current": current,
            "num_violations": len(self._conservation_violations),
            "violations": self._conservation_violations[-10:],  # Last 10
            "total_energy_input": self._total_energy_input,
        }
    
    def _record_energy_input(self, amount: float) -> None:
        """Record energy entering the system from data."""
        self._total_energy_input += amount

    def _idle_ready(self) -> bool:
        return (time.time() - self._last_activity_time) >= float(self.cfg.idle_think_delay_seconds)

    def total_energy(self) -> torch.Tensor:
        """Homeostasis energy for semantic dynamics.

        Excitation is treated as the dominant energetic quantity for the grammar.
        """
        total = super().total_energy()
        exc = self.attractors.get("excitation")
        return total + exc.abs().sum().to(torch.float32)

    @staticmethod
    def _init_embeddings(vocab_size: int, embed_dim: int, *, eps: float, device: torch.device) -> torch.Tensor:
        if embed_dim >= vocab_size:
            emb = torch.eye(vocab_size, embed_dim, device=device, dtype=torch.float32)
        else:
            emb = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float32)
            emb = emb / (emb.norm(dim=1, keepdim=True) + eps)
        return emb

    # ----------------------------
    # Ingest
    # ----------------------------

    def ingest_ids(self, ids: torch.Tensor) -> None:
        """Ingest a token-id context as particles."""
        self._mark_activity()
        ids = ids.to(device=self.device, dtype=torch.long).flatten()
        n = int(ids.numel())
        if n == 0:
            self.particles = BatchState.empty()
            return

        idx = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        energy = idx / (idx.max() + self.cfg.eps)

        pos = self.attractors.get("position")[ids]
        self.particles = BatchState(
            {
                "id": ids,
                "position": pos,
                "energy": energy,
                "heat": torch.zeros(n, device=self.device, dtype=torch.float32),
            }
        )

        # The incoming data IS the energy.
        # Each token carries energy into the system - no normalization.
        # More tokens = more energy entering the system.
        
        # Track total energy input for conservation validation
        energy_input = float(energy.sum().item())
        self._record_energy_input(energy_input)
        
        # Energy: the raw input energy from the data stream
        att_energy = self.attractors.get("energy")
        att_energy.index_add_(0, ids, energy)
        self.attractors.set("energy", att_energy)
        
        self._fresh_observation = True

        # Reset thinking state for the new observation.
        self.halt_mass = 0.0
        self.last_entropy = None
        self.last_confidence = 0.0
        self.last_debug = {}

    def set_diagnostics(self, diagnostics: Optional[SemanticDiagnosticsLogger]) -> None:
        self._diagnostics = diagnostics

    # ----------------------------
    # Idle pondering
    # ----------------------------

    def _active_sources_from_excitation(self) -> torch.Tensor:
        """Select active sources stochastically (Boltzmann-style), no deterministic top-k."""
        exc = self.attractors.get("excitation").clamp(min=0.0)
        heat = self.attractors.get("heat").clamp(min=0.0)

        ids_exc = torch.nonzero(exc > 0, as_tuple=False).flatten()
        ids_heat = torch.nonzero(heat > 0, as_tuple=False).flatten()
        if ids_exc.numel() == 0 and ids_heat.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        eps = self.cfg.eps
        temp = float(self.cfg.dream_sampling_temperature)
        if temp <= 0:
            temp = 1.0

        active: list[torch.Tensor] = []
        if ids_exc.numel() > 0:
            k_exc = max(1, int(round(float(ids_exc.numel()) ** 0.5)))
            k_exc = min(k_exc, int(ids_exc.numel()))
            w = exc[ids_exc].to(torch.float32)
            w = torch.exp(torch.log(w + eps) / temp)
            w = w / (w.sum() + eps)
            pick = torch.multinomial(w, k_exc, replacement=False)
            active.append(ids_exc[pick])
        if ids_heat.numel() > 0:
            k_heat = max(1, int(round(float(ids_heat.numel()) ** 0.5)))
            k_heat = min(k_heat, int(ids_heat.numel()))
            w = heat[ids_heat].to(torch.float32)
            w = torch.exp(torch.log(w + eps) / temp)
            w = w / (w.sum() + eps)
            pick = torch.multinomial(w, k_heat, replacement=False)
            active.append(ids_heat[pick])

        if not active:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return torch.unique(torch.cat(active, dim=0))

    def _update_carrier_baselines(self) -> None:
        """Update slow per-token baselines for excitation and heat."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        tau = float(self.cfg.carrier_tau)
        if tau <= 0:
            tau = 1.0
        alpha = dt / (tau + dt)

        exc = self.attractors.get("excitation").to(torch.float32).clamp(min=0.0)
        heat = self.attractors.get("heat").to(torch.float32).clamp(min=0.0)

        if self._excitation_baseline is None:
            self._excitation_baseline = exc.detach().clone()
        else:
            base = self._excitation_baseline.to(torch.float32)
            self._excitation_baseline = (1.0 - alpha) * base + alpha * exc.detach()

        if self._heat_baseline is None:
            self._heat_baseline = heat.detach().clone()
        else:
            base = self._heat_baseline.to(torch.float32)
            self._heat_baseline = (1.0 - alpha) * base + alpha * heat.detach()

        self._excitation_baseline = self._excitation_baseline.clamp(min=eps)
        self._heat_baseline = self._heat_baseline.clamp(min=eps)

    def predict_from_token(self, token_id: int) -> torch.Tensor:
        """Sparse next-token proposal conditioned on a single token."""
        eps = self.cfg.eps
        out = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        if self.graph.num_edges == 0:
            return out
        src = torch.tensor([int(token_id)], device=self.device, dtype=torch.long)
        batch = self.graph.batch_edges(src)
        if batch is None:
            return out
        w = batch.w.clamp(min=0.0)
        s = w.sum()
        if float(s.item()) <= float(eps):
            return out
        w_norm = w / (s + eps)
        out.index_add_(0, batch.dst, w_norm.to(out.dtype))
        return out

    def observe_next_token(
        self,
        next_id: int,
        *,
        probs: Optional[torch.Tensor] = None,
        cur_id: Optional[int] = None,
        mass_scale: Optional[torch.Tensor] = None,
        mark_activity: bool = True,
    ) -> Dict[str, float]:
        """Local learning hook usable both online and during dreaming."""
        if mark_activity:
            self._mark_activity()
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        next_i = int(next_id)
        if cur_id is None:
            if self.particles.n == 0:
                return {"surprise": 0.0, "pressure": 0.0}
            cur = int(self.particles.get("id")[-1].item())
        else:
            cur = int(cur_id)

        if probs is None:
            # predict_from_token returns a normalized distribution, not logits
            probs = self.predict_from_token(cur)
            probs_sum = probs.sum()
            if probs_sum > eps:
                probs = probs / probs_sum
            else:
                # No outgoing edges from cur - uniform distribution means max surprise
                probs = torch.ones_like(probs) / probs.numel()

        p = probs[next_i].clamp(min=eps)
        surprise = (-torch.log(p)).to(torch.float32)

        # Surprise baseline (homeostatic scale).
        if self._surprise_baseline is None:
            self._surprise_baseline = surprise.detach().clone()
        else:
            base = self._surprise_baseline
            alpha = dt / (float(self.cfg.tau) + dt)
            self._surprise_baseline = base * (1.0 - alpha) + surprise.detach() * alpha
        base = self._surprise_baseline

        pressure = surprise / (surprise + base + eps)
        # Plasticity gate: pressure in [0,1], used to relax homeostasis during surprise.
        self._plasticity_gate = pressure.detach().clone()

        # Hunger modulates plasticity without hard thresholds.
        h = self.hunger[cur].clamp(min=0.0)
        h_scale = self.hunger.abs().mean() + eps
        hunger_factor = h / (h + h_scale + eps)

        mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * pressure * (1.0 + hunger_factor)
        if mass_scale is not None:
            mass = mass * mass_scale.to(device=self.device, dtype=mass.dtype).clamp(min=0.0)

        src = torch.tensor([cur], device=self.device, dtype=torch.long)
        dst = torch.tensor([next_i], device=self.device, dtype=torch.long)
        self.graph.add_edges(src, dst, mass)

        # The incoming data IS the energy.
        # Each observed token brings a unit of energy into the system.
        # This is independent of surprise - the data itself is the energy source.
        input_energy = 1.0
        
        # Track total energy input for conservation validation
        self._record_energy_input(input_energy)
        
        # Energy enters at the observed (destination) token
        energy = self.attractors.get("energy")
        energy[next_i] = energy[next_i] + input_energy
        self.attractors.set("energy", energy)

        self.last_debug.update({"surprise": float(surprise.item()), "shock": float(pressure.item())})
        return {"surprise": float(surprise.item()), "pressure": float(pressure.item())}

    def _ponder_transitive_closure(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> Dict[str, float]:
        """Infer shortcut bonds A->C from A->B and B->C (sparse 2-hop closure)."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        if active_src.numel() == 0 or self.graph.num_edges == 0:
            return {"shortcuts": 0.0}

        hop1 = self.graph.batch_edges(active_src)
        if hop1 is None:
            return {"shortcuts": 0.0}

        # Normalize hop1 per source (A).
        a_u, inv1 = torch.unique(hop1.src, return_inverse=True)
        out1 = torch.zeros(int(a_u.numel()), device=self.device, dtype=torch.float32)
        out1.index_add_(0, inv1, hop1.w)
        w1 = hop1.w / (out1[inv1] + eps)

        mids = torch.unique(hop1.dst)
        hop2 = self.graph.batch_edges(mids)
        if hop2 is None:
            return {"shortcuts": 0.0}

        # Normalize hop2 per source (B).
        b_u, inv2 = torch.unique(hop2.src, return_inverse=True)
        out2 = torch.zeros(int(b_u.numel()), device=self.device, dtype=torch.float32)
        out2.index_add_(0, inv2, hop2.w)
        w2 = hop2.w / (out2[inv2] + eps)

        # Join on B with a bounded per-mid loop (mid count is locality-derived).
        order1 = torch.argsort(hop1.dst)
        dst1 = hop1.dst[order1]
        src1 = hop1.src[order1]
        w1s = w1[order1]
        u1, c1 = torch.unique_consecutive(dst1, return_counts=True)
        s1 = torch.cumsum(torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), c1[:-1]]), dim=0)

        order2 = torch.argsort(hop2.src)
        src2 = hop2.src[order2]
        dst2 = hop2.dst[order2]
        w2s = w2[order2]
        u2, c2 = torch.unique_consecutive(src2, return_counts=True)
        s2 = torch.cumsum(torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), c2[:-1]]), dim=0)

        # Intersection of mids present in both.
        pos = torch.searchsorted(u2, u1)
        in_range = pos < u2.numel()
        pos_safe = pos.clamp(max=max(int(u2.numel()) - 1, 0))
        hit = in_range & (u2[pos_safe] == u1)
        mids_i = u1[hit]
        if mids_i.numel() == 0:
            return {"shortcuts": 0.0}

        pos2 = pos_safe[hit]
        i1 = torch.nonzero(hit, as_tuple=False).flatten()

        src_all: list[torch.Tensor] = []
        dst_all: list[torch.Tensor] = []
        mass_all: list[torch.Tensor] = []

        emb = self.attractors.get("position")

        for j in range(int(mids_i.numel())):
            # hop1 slice for this mid
            start1 = int(s1[i1[j]].item())
            end1 = start1 + int(c1[i1[j]].item())
            a = src1[start1:end1]
            w_a = w1s[start1:end1]

            # hop2 slice for this mid
            start2 = int(s2[pos2[j]].item())
            end2 = start2 + int(c2[pos2[j]].item())
            c = dst2[start2:end2]
            w_c = w2s[start2:end2]

            if a.numel() == 0 or c.numel() == 0:
                continue

            # Cartesian product A×C for this mid.
            aa = a.repeat_interleave(int(c.numel()))
            cc = c.repeat(int(a.numel()))
            path_mass = w_a.repeat_interleave(int(c.numel())) * w_c.repeat(int(a.numel()))

            # Geometric compatibility (derived): exp(-d2 / mean(d2)).
            pa = emb[aa]
            pc = emb[cc]
            diff = pa - pc
            d2 = (diff * diff).sum(dim=1)
            d2_scale = d2.mean() + eps
            geom = torch.exp(-d2 / d2_scale)

            mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * path_mass * geom

            src_all.append(aa)
            dst_all.append(cc)
            mass_all.append(mass)

        if not src_all:
            return {"shortcuts": 0.0}

        src_cat = torch.cat(src_all, dim=0)
        dst_cat = torch.cat(dst_all, dim=0)
        mass_cat = torch.cat(mass_all, dim=0)

        # Novelty: only add missing edges (no hard thresholds).
        _, _, exists = self.graph.get_edges(src_cat, dst_cat)
        add = ~exists
        if not bool(add.any()):
            return {"shortcuts": 0.0}

        self.graph.add_edges(src_cat[add], dst_cat[add], mass_cat[add])

        # Heat release on sources that formed shortcuts.
        heat = self.attractors.get("heat")
        heat.index_add_(0, src_cat[add], mass_cat[add].to(heat.dtype))
        self.attractors.set("heat", heat)

        return {"shortcuts": float(int(add.sum().item()))}

    def _ponder_conflict_resolution(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> Dict[str, float]:
        """Resolve ambiguity by focusing competition on high-entropy sources."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        if active_src.numel() == 0 or self.graph.num_edges == 0:
            return {"confused": 0.0}

        batch = self.graph.batch_edges(active_src)
        if batch is None:
            return {"confused": 0.0}

        s_u, inv = torch.unique(batch.src, return_inverse=True)
        out_sum = torch.zeros(int(s_u.numel()), device=self.device, dtype=torch.float32)
        out_sum.index_add_(0, inv, batch.w)
        w = batch.w / (out_sum[inv] + eps)

        ent_e = -(w * torch.log(w + eps))
        ent = torch.zeros(int(s_u.numel()), device=self.device, dtype=torch.float32)
        ent.index_add_(0, inv, ent_e)

        med = ent.median()
        confused = s_u[ent >= med]
        if confused.numel() == 0:
            return {"confused": 0.0}

        exc = self.attractors.get("excitation")
        ex = exc[confused]
        scale = ex.abs().mean() + eps
        ent_scale = ent[ent >= med].mean() + eps
        noise = torch.randn_like(ex) * (scale * (ent[ent >= med] / ent_scale))
        exc.index_add_(0, confused, torch.tensor(dt, device=self.device) * noise.to(exc.dtype))
        exc = exc.clamp(min=0.0)
        self.attractors.set("excitation", exc)

        # Let competition play out thermodynamically.
        self.run_metabolism(confused, ratio=ratio)
        self.propagate_flow(confused, ratio=ratio)
        return {"confused": float(int(confused.numel()))}

    def _ponder_dream_rollouts(self, active_src: torch.Tensor, *, ratio: torch.Tensor, steps: int) -> Dict[str, float]:
        """Hypothetical rollouts to find dead-ends and build hunger signals."""
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        exc = self.attractors.get("excitation").clamp(min=0.0)
        if active_src.numel() == 0:
            return {"dreams": 0.0, "dead_ends": 0.0}

        # Energy/heat budget loop (state-dependent cognitive depth).
        budget = torch.tensor(float(self.cfg.dream_energy_budget), device=self.device, dtype=torch.float32).clamp(min=0.0)
        stress = (ratio.to(torch.float32) - 1.0).clamp(min=0.0)
        budget = budget * (1.0 + float(self.cfg.dream_budget_stress_gain) * stress)

        temp = float(self.cfg.dream_sampling_temperature)
        if temp <= 0:
            temp = 1.0
        w = exc[active_src].to(torch.float32)
        w = torch.exp(torch.log(w + eps) / temp)
        w = w / (w.sum() + eps)

        dead_ends = 0
        rollouts = 0
        heat = self.attractors.get("heat").clamp(min=0.0).to(torch.float32)
        while float(budget.item()) > 0.0:
            cur = int(active_src[int(torch.multinomial(w, 1).item())].item())
            cum_prob = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            for _t in range(int(steps)):
                if float(self.cfg.dream_entropy_stop) > 0.0:
                    ent = float(self.entropy().item())
                    if ent <= float(self.cfg.dream_entropy_stop):
                        budget = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                        break
                dist = self.predict_from_token(cur)
                s = float(dist.abs().sum().item())
                if s <= float(eps):
                    self.hunger[cur] = self.hunger[cur] + torch.tensor(dt, device=self.device, dtype=self.hunger.dtype)
                    dead_ends += 1
                    break
                probs = dist / (dist.sum() + eps)
                nxt = int(torch.multinomial(probs, 1).item())
                cum_prob = cum_prob * probs[nxt].to(cum_prob.dtype)
                self.observe_next_token(
                    nxt,
                    probs=probs,
                    cur_id=cur,
                    mass_scale=cum_prob.detach(),
                    mark_activity=False,
                )
                cur = nxt
                heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else (heat.abs().mean() + eps)).to(torch.float32)
                step_cost = 1.0 + (heat[cur] / (heat_scale + eps)).clamp(min=0.0)
                budget = budget - step_cost
                if float(budget.item()) <= 0.0:
                    break
            rollouts += 1

        # Hunger decays homeostatically (no manual resets).
        h_scale = self.hunger.abs().mean() + eps
        self.hunger = self.hunger * torch.exp(-torch.tensor(dt, device=self.device) * ratio.to(self.hunger.dtype) / h_scale)
        return {"dreams": float(rollouts), "dead_ends": float(dead_ends)}

    def idle_think(self, *, steps: int = 1, dream_steps: int = 8) -> Dict[str, float]:
        """Idle pondering: discover relations from existing structure (no external data)."""
        if not self._idle_ready():
            return {}
        steps = int(steps)
        dream_steps = int(dream_steps)
        if steps <= 0:
            return {}

        self._update_carrier_baselines()
        ratio = self._homeostasis_ratio(plasticity_gate=self._plasticity_gate).to(torch.float32)
        active = self._active_sources_from_excitation()

        agg: Dict[str, float] = {}
        for _ in range(steps):
            r1 = self._ponder_transitive_closure(active, ratio=ratio)
            r2 = self._ponder_conflict_resolution(active, ratio=ratio)
            r3 = self._ponder_dream_rollouts(active, ratio=ratio, steps=dream_steps)
            agg.update({k: agg.get(k, 0.0) + float(v) for k, v in {**r1, **r2, **r3}.items()})

        if self._diagnostics is not None:
            self._diagnostics.log(
                {
                    "step": int(self._ponder_step),
                    "active": int(active.numel()),
                    "shortcuts": float(agg.get("shortcuts", 0.0)),
                    "confused": float(agg.get("confused", 0.0)),
                    "dreams": float(agg.get("dreams", 0.0)),
                    "dead_ends": float(agg.get("dead_ends", 0.0)),
                    "hunger_mean": float(self.hunger.mean().item()),
                    "heat_mean": float(self.attractors.get("heat").mean().item()),
                    "exc_mean": float(self.attractors.get("excitation").mean().item()),
                }
            )
        self._ponder_step += 1
        return agg

    # ----------------------------
    # Grammar phases
    # ----------------------------

    def update_topology(self) -> None:
        """Topology update phase: nucleate/strengthen edges from observed order."""
        if self.particles.n < 2:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        ids = self.particles.get("id")
        e = self.particles.get("energy")
        e_scale = e.abs().mean() + eps

        # Nucleation mass emerges from current context energy scale.
        mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * (e.mean() / e_scale)
        self.graph.add_path(ids, mass)

    def run_metabolism(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Metabolism phase: update bond masses and traces for active sources."""
        if active_src.numel() == 0 or self.graph.num_edges == 0:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        exc = self.attractors.get("excitation")
        heat = self.attractors.get("heat")
        exc_scale = exc[active_src].abs().mean() + eps

        batch = self.graph.batch_edges(active_src)
        if batch is None:
            return

        # Local homeostasis per source token (carrier baseline).
        if self._excitation_baseline is not None:
            base_src = self._excitation_baseline[batch.src].to(torch.float32)
            local = (torch.log1p(exc[batch.src].to(torch.float32).clamp(min=0.0)) / (torch.log1p(base_src) + eps)).clamp(min=eps)
            ratio_src = ratio.to(torch.float32) * local
        else:
            ratio_src = ratio.to(torch.float32)

        # Outgoing normalization per source (active subset only).
        src_u, inv = torch.unique(batch.src, return_inverse=True)
        out_sum = torch.zeros(int(src_u.numel()), device=self.device, dtype=torch.float32)
        out_sum.index_add_(0, inv, batch.w)
        w_norm = batch.w / (out_sum[inv] + eps)

        # Flow-based usage.
        use = exc[batch.src] * w_norm

        # Heat reduces usable energy (entropy).
        h_level = heat[batch.src].abs().mean()
        heat_utility = 1.0 / (1.0 + h_level + eps)
        income = use * heat_utility

        # Trace: local time-extended credit.
        trace = batch.trace
        trace_scale = trace.abs().mean() + eps
        trace_decay = torch.exp(-dt * ratio_src.to(trace.dtype) / (exc_scale + trace_scale))
        trace_new = trace * trace_decay + income

        # Cost: proportional decay (edges starve without use).
        cost = ratio_src.to(batch.w.dtype) * exc_scale * batch.w / (batch.w.abs().mean() + eps)
        w_new = (batch.w + dt * (income - cost)).clamp(min=0.0)

        # Write back.
        self.graph.update_edges(batch.eidx, w_new, trace_new)

        # Prune weak edges without fixed thresholds.
        self.graph.prune_by_src_mean(active_src)
        self.graph.compact()

        self.last_debug.update(
            {
                "exc_scale": float(exc_scale.item()),
                "heat_level": float(h_level.item()),
                "income_mean": float(income.mean().item()),
                "cost_mean": float(cost.mean().item()),
                "edges_active": float(int(batch.eidx.numel())),
            }
        )

    def propagate_flow(self, active_src: torch.Tensor, *, ratio: torch.Tensor) -> None:
        """Thermodynamic flow propagation.
        
        Heat is the transport mechanism. The cascade:
        1. Heat arrives at entity
        2. Energy transfers with heat
        3. Temperature rises (kinetic energy)
        4. Excitation rises with temperature
        5. Excitation generates heat
        6. Heat flows out via bonds (split by attraction/gravity)
        
        This repeats for particles → bonds → carriers → particles...
        """
        if active_src.numel() == 0:
            return
        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        # Get current state
        p_energy = self.attractors.get("energy")
        p_heat = self.attractors.get("heat")
        p_exc = self.attractors.get("excitation")
        positions = self.attractors.get("position")
        
        c_energy = self.carriers.energy
        c_heat = self.carriers.heat
        
        # Temperature = heat / mass (assume unit mass)
        p_temp = p_heat.clone()
        c_temp = c_heat.clone()
        
        # ========================================
        # Step 1: Active particles bond to carriers
        # ========================================
        active_positions = positions[active_src]
        carrier_indices, attractions = self.carriers.compute_attractions(
            active_positions, 
            top_k=self.bonds_per_particle
        )
        
        n_active = active_src.numel()
        k = carrier_indices.shape[1]
        particle_ids_flat = active_src.unsqueeze(1).expand(-1, k).reshape(-1)
        carrier_ids_flat = carrier_indices.reshape(-1)
        attractions_flat = attractions.reshape(-1)
        
        # Bond strength = attraction (gravity-like)
        self.particle_carrier_bonds.add_bonds(
            particle_ids_flat, 
            carrier_ids_flat, 
            attractions_flat * dt
        )
        
        # ========================================
        # Step 2: Particle thermodynamics
        # ========================================
        # Temperature rises with heat
        # Excitation rises with temperature
        p_exc = p_exc + p_temp * dt
        
        # Excitation generates heat (excitation is consumed)
        heat_generated = p_exc * dt
        p_heat = p_heat + heat_generated
        p_exc = p_exc - heat_generated
        
        # ========================================
        # Step 3: Heat flows from particles to carriers
        # ========================================
        # Heat flows via bonds, split by attraction (gravity)
        if self.particle_carrier_bonds.num_bonds > 0:
            # Heat to flow out = fraction of particle heat
            heat_to_flow = p_heat * dt
            
            # Flow heat from particles to carriers (tracks last_energy_flow)
            carrier_heat_in = self.particle_carrier_bonds.flow_particle_to_carriers(heat_to_flow)
            
            # Compute per-particle heat outflow
            p_ids = self.particle_carrier_bonds.particle_ids
            strengths = self.particle_carrier_bonds.strengths
            
            p_total_strength = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
            p_total_strength.index_add_(0, p_ids, strengths)
            
            bond_ratios = strengths / (p_total_strength[p_ids] + eps)
            bond_heat_out = heat_to_flow[p_ids] * bond_ratios
            
            heat_out_per_particle = torch.zeros_like(p_heat)
            heat_out_per_particle.index_add_(0, p_ids, bond_heat_out)
            
            # Particles lose heat, carriers gain heat
            p_heat = p_heat - heat_out_per_particle
            c_heat = c_heat + carrier_heat_in
            
            # Energy transfers WITH heat (heat carries energy)
            # Energy ratio = heat ratio
            energy_to_flow = p_energy * dt
            carrier_energy_in = self.particle_carrier_bonds.flow_particle_to_carriers(energy_to_flow)
            
            bond_energy_out = energy_to_flow[p_ids] * bond_ratios
            energy_out_per_particle = torch.zeros_like(p_energy)
            energy_out_per_particle.index_add_(0, p_ids, bond_energy_out)
            
            p_energy = p_energy - energy_out_per_particle
            c_energy = c_energy + carrier_energy_in
        
        # ========================================
        # Step 4: Carrier thermodynamics
        # ========================================
        c_temp = c_heat.clone()
        c_exc = c_temp * dt  # Carriers get excited by temperature
        
        # Carrier excitation generates heat
        c_heat_generated = c_exc * dt
        c_heat = c_heat + c_heat_generated
        
        # ========================================
        # Step 5: Heat flows from carriers to particles
        # ========================================
        if self.particle_carrier_bonds.num_bonds > 0:
            # Heat flows to ALL bonded particles (not just non-sources)
            heat_to_distribute = c_heat * dt
            
            particle_heat_in = self.particle_carrier_bonds.flow_carriers_to_particles(
                heat_to_distribute,
                exclude_particles=None  # Heat flows to everyone
            )
            
            # Carriers lose heat
            c_ids = self.particle_carrier_bonds.carrier_ids
            c_total_strength = torch.zeros(self.carriers.num_carriers, device=self.device, dtype=torch.float32)
            c_total_strength.index_add_(0, c_ids, self.particle_carrier_bonds.strengths)
            
            # Approximate: carriers lose what they distributed
            c_heat = c_heat - heat_to_distribute * (c_total_strength > eps).float()
            p_heat = p_heat + particle_heat_in
            
            # Energy transfers with heat
            energy_to_distribute = c_energy * dt
            particle_energy_in = self.particle_carrier_bonds.flow_carriers_to_particles(
                energy_to_distribute,
                exclude_particles=None
            )
            
            c_energy = c_energy - energy_to_distribute * (c_total_strength > eps).float()
            p_energy = p_energy + particle_energy_in
        
        # ========================================
        # Snap dead bonds (no heat/energy flow = bond breaks)
        # ========================================
        num_snapped = self.particle_carrier_bonds.snap_dead_bonds()
        
        # Physical constraints (non-negative)
        p_energy = p_energy.clamp(min=0.0)
        p_heat = p_heat.clamp(min=0.0)
        p_exc = p_exc.clamp(min=0.0)
        c_energy = c_energy.clamp(min=0.0)
        c_heat = c_heat.clamp(min=0.0)

        # Save particle state
        self.attractors.set("excitation", p_exc)
        self.attractors.set("heat", p_heat)
        self.attractors.set("energy", p_energy)
        
        # Save carrier state
        self.carriers.energy = c_energy
        self.carriers.heat = c_heat
        
        # Debug info
        self.last_debug.update({
            "num_carriers": self.carriers.num_carriers,
            "num_bonds": self.particle_carrier_bonds.num_bonds,
            "bonds_snapped": num_snapped,
            "carrier_energy": float(c_energy.sum().item()),
            "carrier_heat": float(c_heat.sum().item()),
        })

        # Track total energy (particles + carriers, energy + heat)
        total = p_energy.sum() + p_heat.sum() + c_energy.sum() + c_heat.sum()
        
        # Temperature = average heat (assuming unit mass)
        p_temperature = p_heat.mean()
        
        # Conservation validation
        conservation = self.validate_conservation()
        
        self.last_debug.update(
            {
                "exc_mean": float(p_exc.mean().item()),
                "heat_mean": float(p_heat.mean().item()),
                "energy_mean": float(p_energy.mean().item()),
                "temperature_mean": float(p_temperature.item()),
                "total_energy": float(total.item()),
                "energy_input": self._total_energy_input,
                "conservation_error": conservation["relative_error"],
                "conservation_valid": 1.0 if conservation["valid"] else 0.0,
            }
        )

    def step_grammar(self) -> None:
        """One grammar step, decomposed into phases for debuggability."""
        if self.particles.n == 0:
            return

        self._update_carrier_baselines()
        ratio = self._homeostasis_ratio(plasticity_gate=self._plasticity_gate).to(torch.float32)

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        # Continuous observation: after an ingestion event, don't immediately double-inject.
        if self._fresh_observation:
            self._fresh_observation = False
        else:
            exc = self.attractors.get("excitation")
            ids = self.particles.get("id")
            e = self.particles.get("energy").clamp(min=0.0)
            exc.index_add_(0, ids, torch.tensor(dt, device=self.device) * (e / (e.sum() + eps)))
            self.attractors.set("excitation", exc)

        # Phase 1: topology
        self.update_topology()

        # Active sources: context tokens (locality without global thresholds).
        active_src = torch.unique(self.particles.get("id"))

        # Phase 2: metabolism
        self.run_metabolism(active_src, ratio=ratio)

        # Phase 3: propagation
        self.propagate_flow(active_src, ratio=ratio)

        ent = float(self.entropy().item())
        if self.last_entropy is None:
            self.last_entropy = ent
        self.last_debug["entropy"] = ent
        self.last_confidence = self.thinking_confidence()

    # ----------------------------
    # Readout
    # ----------------------------

    def _context_distribution(self) -> torch.Tensor:
        eps = self.cfg.eps
        dist = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        ids = self.particles.get("id")
        e = self.particles.get("energy").clamp(min=0.0)
        dist.index_add_(0, ids, e / (e.sum() + eps))
        return dist

    def predict_next(self) -> torch.Tensor:
        eps = self.cfg.eps
        ctx = self._context_distribution()
        gram = self.graph.flow_from_distribution(ctx)

        c_scale = ctx.abs().sum() + eps
        g_scale = gram.abs().sum()

        # Readout prioritizes *forward* flow when grammar has support; otherwise fall back to context.
        if float(g_scale.item()) > float(eps):
            return gram / (g_scale + eps)
        return ctx / c_scale

    def entropy(self) -> torch.Tensor:
        """Compute entropy of the prediction distribution.
        
        Note: predict_next() returns a normalized distribution, not logits.
        We compute entropy directly from that distribution.
        """
        probs = self.predict_next()
        # Ensure it's a valid probability distribution
        probs = probs.clamp(min=0)
        prob_sum = probs.sum()
        if prob_sum > self.cfg.eps:
            probs = probs / prob_sum
        else:
            # Fallback to uniform if no probability mass
            return torch.tensor(0.0, device=self.device, dtype=probs.dtype)
        
        # Entropy: -sum(p * log(p)), only for non-zero probabilities
        # Only count tokens with actual probability mass
        active_mask = probs > self.cfg.eps
        active_probs = probs[active_mask]
        
        if active_probs.numel() == 0:
            return torch.tensor(0.0, device=self.device, dtype=probs.dtype)
        
        log_probs = torch.log(active_probs)
        entropy = -(active_probs * log_probs).sum()
        
        # Store debug info about distribution shape
        self.last_debug["num_active_probs"] = float(active_probs.numel())
        self.last_debug["max_prob"] = float(active_probs.max().item())
        self.last_debug["min_active_prob"] = float(active_probs.min().item())
        
        return entropy

    def thinking_confidence(self) -> float:
        ctx = self._context_distribution()
        gram = self.graph.flow_from_distribution(ctx)
        c = ctx.abs().sum() + self.cfg.eps
        g = gram.abs().sum()
        # Confidence is the amount of forward signal available relative to the observation mass.
        conf = (g / (c + self.cfg.eps)).clamp(min=0.0, max=1.0)
        return float(conf.item())

    def thinking_complete(self) -> bool:
        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        conf = self.thinking_confidence()
        self.halt_mass = min(1.0, self.halt_mass + (conf * dt) / (conf + eps))
        return self.halt_mass >= 1.0

    def output_state(self) -> SemanticOutput:
        # predict_next() returns a normalized distribution, not logits
        probs = self.predict_next()
        probs = probs.clamp(min=0)
        prob_sum = probs.sum()
        if prob_sum > self.cfg.eps:
            probs = probs / prob_sum
        
        # For logits, convert probs back to log-odds (useful for some downstream uses)
        logits = torch.log(probs + self.cfg.eps)
        
        idx = int(torch.argmax(probs).item())
        tok = self.vocab[idx] if 0 <= idx < len(self.vocab) else None
        meta = {
            "entropy": float(self.entropy().item()),
            "confidence": float(self.last_confidence),
            **self.last_debug,
        }
        return SemanticOutput(logits=logits, probs=probs, token_index=idx, token=tok, meta=meta)
