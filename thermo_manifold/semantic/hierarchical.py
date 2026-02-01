from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from .manifold import SemanticManifold, SemanticOutput
from .chunk_store import ChunkStore
from .bipartite_graph import SparseBipartiteBondGraph


class HierarchicalSemanticManifold(SemanticManifold):
    """Semantic manifold with hierarchical nucleation (slow "chunk" particles).

    Design goals:
    - No backprop.
    - Online structural re-alignment: bonds and chunks update from ongoing observation.
    - Context is represented by persistent energetic objects (chunks) rather than a fixed
      attention window.

    The hierarchy implemented here is minimal but functional:
    - Token layer: token->token sparse bond graph (inherits from SemanticManifold)
    - Chunk layer: trigram chunks (order=3) with chunk->token sparse bipartite bonds

    The chunk layer is created and reinforced via:
    - Coherence (binding energy) of token bonds (condensation)
    - Metabolic shock from prediction surprise (error-driven nucleation)
    """

    def __init__(
        self,
        config,
        device: torch.device,
        *,
        vocab: list[str],
        embed_dim: Optional[int] = None,
        chunk_min_len: int = 2,
        chunk_max_len: int = 4,
    ):
        super().__init__(config=config, device=device, vocab=vocab, embed_dim=embed_dim)

        self.chunk_min_len = int(chunk_min_len)
        self.chunk_max_len = int(chunk_max_len)
        if self.chunk_min_len < 2:
            self.chunk_min_len = 2
        if self.chunk_max_len < self.chunk_min_len:
            self.chunk_max_len = self.chunk_min_len

        self.chunks = ChunkStore(
            order=self.chunk_max_len,
            vocab_size=self.vocab_size,
            sem_dim=self.embed_dim,
            device=device,
            eps=self.cfg.eps,
        )
        self.chunk_graph = SparseBipartiteBondGraph(
            num_src=0,
            num_dst=self.vocab_size,
            device=device,
            dtype=torch.float32,
            eps=self.cfg.eps,
        )

        # Surprise baseline for error-driven nucleation.
        self._surprise_baseline: Optional[torch.Tensor] = None

        # Cache last probs for convenience.
        self._last_probs: Optional[torch.Tensor] = None

    # ----------------------------
    # Homeostasis energy
    # ----------------------------

    def total_energy(self) -> torch.Tensor:
        total = super().total_energy()

        if self.chunks.num_chunks > 0:
            total = total + self.chunks.energy.abs().sum().to(torch.float32)
            total = total + self.chunks.excitation.abs().sum().to(torch.float32)

        if self.chunk_graph.num_edges > 0:
            total = total + self.chunk_graph.w.abs().sum().to(torch.float32)

        return total

    # ----------------------------
    # Chunk helpers
    # ----------------------------

    def _recent_seqs(self) -> list[torch.Tensor]:
        """Return recent sequence candidates of multiple lengths."""
        ids = self.particles.get("id")
        n = int(ids.numel())
        out: list[torch.Tensor] = []
        for L in range(self.chunk_min_len, self.chunk_max_len + 1):
            if n >= L:
                out.append(ids[-L:].view(1, -1))
        return out

    def _binding(self, seq: torch.Tensor) -> torch.Tensor:
        """Binding energy density proxy for a variable-length seq."""

        eps = self.cfg.eps
        if self.graph.num_edges == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim != 2 or int(seq.shape[0]) != 1 or int(seq.shape[1]) < 2:
            raise ValueError("seq must be [1,L] with L>=2")
        s = seq[0]
        src = s[:-1].contiguous()
        dst = s[1:].contiguous()

        w, tr, _ = self.graph.get_edges(src, dst)

        w_scale = self.graph.w.abs().mean() + eps
        t_scale = self.graph.trace.abs().mean() + eps

        w_n = w.to(torch.float32) / w_scale
        t_n = tr.to(torch.float32) / t_scale

        bond = torch.sqrt((w_n * t_n).clamp(min=0.0) + eps)
        # Energy density: geometric mean across edges (length-normalized).
        return torch.exp(torch.log(bond + eps).mean())

    def _best_candidate(self) -> Optional[torch.Tensor]:
        """Select the best recent sequence candidate by thermodynamic bidding."""
        eps = self.cfg.eps
        dt = float(self.cfg.dt)

        best_seq: Optional[torch.Tensor] = None
        best_score = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        for seq in self._recent_seqs():
            L = int(seq.shape[1])
            binding = self._binding(seq)
            base = self.chunks.update_binding_baseline(binding, length=L, dt=dt)
            pressure = (binding - base).clamp(min=0.0) / (binding + base + eps)
            score = pressure * binding
            if bool(score > best_score):
                best_score = score
                best_seq = seq
        return best_seq

    def _chunk_condensation(self, *, ratio: torch.Tensor) -> None:
        """Condense and activate a chunk from the current context (multi-resolution)."""

        seq = self._best_candidate()
        if seq is None:
            return

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        binding = self._binding(seq)
        base = self.chunks.update_binding_baseline(binding, length=int(seq.shape[1]), dt=dt)

        # Condensation pressure: supersaturation (binding above baseline).
        pressure = (binding - base).clamp(min=0.0) / (binding + base + eps)
        mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * pressure

        # Ensure chunk exists when there is condensation mass.
        res = self.chunks.add_or_reinforce(seq, mass, word_pos=self.attractors.get('position'))
        if self.chunks.num_chunks != self.chunk_graph.num_src:
            self.chunk_graph.set_num_src(self.chunks.num_chunks)

        # Always activate existing chunk when observed (even if no condensation).
        act = torch.tensor(dt, device=self.device, dtype=torch.float32) * (binding / (binding + base + eps))
        if bool(res.exists.any()):
            idx = res.idx[res.exists]
            self.chunks.excitation[idx] = self.chunks.excitation[idx] + act

        self.last_debug.update(
            {
                "chunk_binding": float(binding.item()),
                "chunk_baseline": float(base.item()),
                "chunks": float(self.chunks.num_chunks),
            }
        )

    def _chunk_metabolism(self, *, ratio: torch.Tensor) -> None:
        """Decay chunk reservoirs and chunk->token bonds."""

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        # Chunk reservoirs
        self.chunks.decay(ratio=ratio, dt=dt)

        # Chunk bonds
        if self.chunk_graph.num_edges == 0 or self.chunks.num_chunks == 0:
            return

        active = torch.nonzero(self.chunks.excitation > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            return

        batch = self.chunk_graph.batch_edges(active)
        if batch is None:
            return

        w = batch.w
        tr = batch.trace

        # Heat modulates decay: hotter => faster forgetting.
        h = self.chunks.heat
        heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else (h.abs().mean() + eps)).to(torch.float32)
        heat_level = (h.abs().mean().to(torch.float32) / (heat_scale + eps)).clamp(min=0.0)
        heat_factor = 1.0 + heat_level

        decay = torch.exp(-dt * ratio.to(w.dtype) * heat_factor.to(w.dtype))
        w_new = w * decay
        tr_new = tr * decay.to(tr.dtype)

        self.chunk_graph.update_edges(batch.eidx, w_new, tr_new)
        self.chunk_graph.compact()

    def _chunk_to_tokens(self) -> None:
        """Project the most-recent chunk into token excitation.

        Using only the *current* chunk avoids global mixing of old narratives.
        """

        if self.chunks.num_chunks == 0 or self.chunk_graph.num_edges == 0:
            return

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        best_idx: Optional[int] = None
        best_x = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for seq in self._recent_seqs():
            lr = self.chunks.lookup(seq)
            if not bool(lr.exists.any()):
                continue
            cidx = int(lr.idx[lr.exists][0].item())
            x = self.chunks.excitation[cidx]
            if bool(x > best_x):
                best_x = x
                best_idx = cidx
        if best_idx is None:
            return

        dist = torch.zeros(self.chunks.num_chunks, device=self.device, dtype=torch.float32)
        dist[best_idx] = 1.0

        flow = self.chunk_graph.flow_from_distribution(dist)
        if float(flow.abs().sum().item()) <= eps:
            return

        exc = self.attractors.get("excitation")
        exc = exc + torch.tensor(dt, device=self.device, dtype=exc.dtype) * flow.to(exc.dtype)
        self.attractors.set("excitation", exc)

        self.last_debug.update({"chunk_flow_sum": float(flow.sum().item())})

    # ----------------------------
    # Grammar step
    # ----------------------------

    def run_metabolism(self, active_src: torch.Tensor, ratio: torch.Tensor) -> None:
        """Homeostatic decay for word-level bonds.

        Reinforcement comes from observation (update_topology/observe_next_token).
        This metabolism step applies thermostatic decay only to prevent
        self-reinforcement loops from dominating adaptation.
        """

        if active_src.numel() == 0:
            return

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        batch = self.graph.batch_edges(active_src)
        if batch is None:
            return

        heat = self.attractors.get("heat")
        # Per-carrier ratio (local baselines) + per-edge heat modulation.
        if self._excitation_baseline is not None:
            exc = self.attractors.get("excitation")
            base_src = self._excitation_baseline[batch.src].to(torch.float32)
            local = (torch.log1p(exc[batch.src].to(torch.float32).clamp(min=0.0)) / (torch.log1p(base_src) + eps)).clamp(min=eps)
            ratio_src = ratio.to(torch.float32) * local
        else:
            ratio_src = ratio.to(torch.float32)

        heat_scale = (self._heat_abs_scale if self._heat_abs_scale is not None else (heat.abs().mean() + eps)).to(torch.float32)
        heat_factor = 1.0 + (heat[batch.src].to(torch.float32).clamp(min=0.0) / (heat_scale + eps))

        w = batch.w
        tr = batch.trace

        w_decay = torch.exp(-dt * ratio_src.to(w.dtype) * heat_factor.to(w.dtype))
        t_decay = torch.exp(-dt * ratio_src.to(tr.dtype) * heat_factor.to(tr.dtype))

        w_new = w * w_decay
        tr_new = tr * t_decay

        self.graph.update_edges(batch.eidx, w_new, tr_new)
        self.graph.compact()


    def propagate_flow(self, active_src: torch.Tensor, ratio: torch.Tensor) -> None:
        """Propagate grammatical flow and update attractor excitation/heat.

        This override uses *direct* scale-dependent decay (large excitation => faster decay),
        which improves energetic homeostasis during continuous streaming.
        """

        if active_src.numel() == 0:
            return

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        ctx = self._context_distribution()
        flow = self.graph.flow_from_distribution(ctx)
        f_scale = flow.abs().mean() + eps
        flow = flow / f_scale

        exc = self.attractors.get("excitation")
        exc_scale = exc.abs().mean() + eps
        if self._excitation_baseline is not None:
            base = self._excitation_baseline.to(torch.float32)
            local = (torch.log1p(exc.to(torch.float32).clamp(min=0.0)) / (torch.log1p(base) + eps)).clamp(min=eps)
            ratio_vec = ratio.to(torch.float32) * local
        else:
            ratio_vec = ratio.to(torch.float32)

        exc_decay = torch.exp(-dt * ratio_vec.to(exc.dtype) * exc_scale)
        exc_new = exc * exc_decay + flow
        self.attractors.set("excitation", exc_new)

        heat = self.attractors.get("heat")
        heat_scale = heat.abs().mean() + eps
        heat_decay = torch.exp(-dt * ratio_vec.to(heat.dtype) * heat_scale)
        heat_utility = 1.0 / (1.0 + heat_scale + eps)
        heat_new = heat * heat_decay + dt * flow.abs() * heat_utility
        self.attractors.set("heat", heat_new)

    def step_grammar(self) -> None:
        if self.particles.n == 0:
            return

        self._update_carrier_baselines()
        ratio = self._homeostasis_ratio(plasticity_gate=self._plasticity_gate).to(torch.float32)

        dt = float(self.cfg.dt)
        eps = self.cfg.eps

        ids = self.particles.get('id')

        # Observation injection: after an ingestion event, don't immediately double-inject.
        if self._fresh_observation:
            self._fresh_observation = False
        else:
            exc = self.attractors.get('excitation')
            e = self.particles.get('energy').clamp(min=0.0)
            exc.index_add_(0, ids, torch.tensor(dt, device=self.device) * (e / (e.sum() + eps)))
            self.attractors.set('excitation', exc)

        # Phase 1: token topology
        self.update_topology()

        # Phase 1b: chunk condensation (hierarchy)
        self._chunk_condensation(ratio=ratio)

        active_src = torch.unique(ids)

        # Phase 2: token metabolism
        self.run_metabolism(active_src, ratio=ratio)

        # Phase 3: token propagation
        self.propagate_flow(active_src, ratio=ratio)

        # Chunk phases: decay + top-down bias
        self._chunk_metabolism(ratio=ratio)
        self._chunk_to_tokens()

        ent = float(self.entropy().item())
        if self.last_entropy is None:
            self.last_entropy = ent
        self.last_debug['entropy'] = ent
        self.last_confidence = self.thinking_confidence()

    # ----------------------------
    # Readout
    # ----------------------------

    def predict_next(self) -> torch.Tensor:
        eps = self.cfg.eps
        ids = self.particles.get("id")
        if ids.numel() == 0:
            return torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)

        # --- Word-level (bigram) proposal: condition primarily on the current token.
        cur = ids[-1].to(torch.long)
        dist_word = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
        dist_word[cur] = 1.0
        word_flow = self.graph.flow_from_distribution(dist_word)
        if word_flow.abs().sum() <= eps:
            # Cold start fallback: use the full context distribution.
            ctx = self._context_distribution()
            word_flow = self.graph.flow_from_distribution(ctx)

        # --- Chunk-level proposal: condition on the most excited recent chunk candidate.
        chunk_flow = torch.zeros_like(word_flow)
        if self.chunks.num_chunks > 0 and self.chunk_graph.num_edges > 0:
            best_idx: Optional[int] = None
            best_x = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for seq in self._recent_seqs():
                lr = self.chunks.lookup(seq)
                if not bool(lr.exists.any()):
                    continue
                cidx = int(lr.idx[lr.exists][0].item())
                x = self.chunks.excitation[cidx]
                if bool(x > best_x):
                    best_x = x
                    best_idx = cidx
            if best_idx is not None:
                dist_chunk = torch.zeros(self.chunks.num_chunks, device=self.device, dtype=torch.float32)
                dist_chunk[best_idx] = 1.0
                chunk_flow = self.chunk_graph.flow_from_distribution(dist_chunk)

        # --- Mixture (no tuned constants): gate by relative mass of the proposals.
        w_mass = word_flow.abs().sum()
        c_mass = chunk_flow.abs().sum()
        gate = c_mass / (c_mass + w_mass + eps)
        combined = (1.0 - gate) * word_flow + gate * chunk_flow
        g_scale = combined.abs().sum()
        if g_scale > eps:
            return combined / (g_scale + eps)
        return dist_word

    def output_state(self) -> SemanticOutput:
        logits = self.predict_next()
        probs = torch.softmax(logits, dim=0)
        self._last_probs = probs.detach()

        idx = int(torch.argmax(probs).item())
        tok = self.vocab[idx] if 0 <= idx < len(self.vocab) else None

        meta = {
            "entropy": float(self.entropy().item()),
            "confidence": float(self.last_confidence),
            **self.last_debug,
        }
        return SemanticOutput(logits=logits, probs=probs, token_index=idx, token=tok, meta=meta)

    # ----------------------------
    # Online learning hook
    # ----------------------------

    def observe_next_token(
        self,
        next_id: Union[int, torch.Tensor],
        *,
        probs: Optional[torch.Tensor] = None,
        cur_id: Optional[int] = None,
        mass_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Observation hook to apply metabolic shock and create new structure.

        This is intentionally local:
        - Compute surprise on the observed next token.
        - Inject mass into the specific transition (token->token) and (chunk->token).

        No gradients, no backprop.
        """

        # Run base token-level reinforcement (also works during dreaming).
        if probs is None:
            probs = self._last_probs
        if probs is None:
            probs = torch.softmax(self.predict_next(), dim=0)
        # Determine the current source token for chunk coupling.
        if cur_id is None:
            if self.particles.n == 0:
                return {"surprise": 0.0, "pressure": 0.0}
            cur = int(self.particles.get("id")[-1].item())
        else:
            cur = int(cur_id)
        out = super().observe_next_token(int(next_id), probs=probs, cur_id=cur, mass_scale=mass_scale)
        # Carry plasticity gate upward so homeostasis can relax during surprise.
        self._plasticity_gate = torch.tensor(float(out.get("pressure", 0.0)), device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        dt = float(self.cfg.dt)
        eps = self.cfg.eps
        # Mirror token-level plasticity for chunk coupling.
        h = self.hunger[cur].clamp(min=0.0)
        h_scale = self.hunger.abs().mean() + eps
        hunger_factor = h / (h + h_scale + eps)
        mass = torch.tensor(dt, device=self.device, dtype=torch.float32) * torch.tensor(out["pressure"], device=self.device) * (1.0 + hunger_factor)
        if mass_scale is not None:
            mass = mass * mass_scale.to(device=self.device, dtype=mass.dtype).clamp(min=0.0)

        # Chunk->token reinforcement using the best multi-resolution candidate.
        seq = self._best_candidate()
        if seq is not None:
            res = self.chunks.add_or_reinforce(seq, mass, word_pos=self.attractors.get('position'))
            if self.chunks.num_chunks != self.chunk_graph.num_src:
                self.chunk_graph.set_num_src(self.chunks.num_chunks)
            if bool(res.exists.any()):
                cidx = res.idx[res.exists]
                dst_c = torch.full_like(cidx, int(next_id), dtype=torch.long)
                m_c = mass.expand_as(cidx)
                self.chunk_graph.add_edges(cidx, dst_c, m_c)

        return out

    def idle_think(self, *, steps: int = 1, dream_steps: int = 8) -> Dict[str, float]:
        """Idle pondering with hierarchy (chunks participate)."""
        out = super().idle_think(steps=steps, dream_steps=dream_steps)
        ratio = self._homeostasis_ratio().to(torch.float32)
        # Let the chunk layer metabolize and bias tokens during idle time.
        self._chunk_metabolism(ratio=ratio)
        self._chunk_to_tokens()
        return out
