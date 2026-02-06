"""Dual-domain inference utilities.

The Sensorium Manifold has two observation domains:
1. Geometric: particles with positions, velocities, energies, heats
2. Coherence: particles coupled to ω-modes via resonance

Inference involves switching between domains to find answers:
- Geometric: for locality, spatial clustering, energy "hotness"
- Coherence: for resonance, phase coherence, mode coupling

This module provides utilities for querying both domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch


@dataclass
class GeometricQuery:
    """Query result from the geometric domain."""
    indices: torch.Tensor          # Particle indices
    positions: torch.Tensor        # (N, 3) positions
    energies: torch.Tensor         # (N,) energies
    heats: torch.Tensor            # (N,) heats
    token_ids: torch.Tensor        # (N,) token IDs
    excitations: torch.Tensor      # (N,) oscillator frequencies


@dataclass
class ModeQuery:
    """Query result from the coherence (ω-mode) domain."""
    mode_indices: torch.Tensor      # Mode indices
    frequencies: torch.Tensor       # Mode ω
    amplitudes: torch.Tensor        # Mode |Ψ|
    phases: torch.Tensor            # Mode arg(Ψ)
    conflict: torch.Tensor          # Mode conflict (lower = more coherent)
    state: torch.Tensor             # 0=volatile, 1=stable, 2=crystallized
    gate_widths: torch.Tensor       # Mode σ (frequency selectivity)


class DualDomainInference:
    """Inference by switching between geometric and coherence domains.
    
    Usage pattern:
    1. Start in one domain (e.g., find "hot" particles)
    2. Switch to other domain (e.g., find modes they couple to)
    3. Switch back (e.g., find other particles coupled to those modes)
    4. Interpret results (e.g., dehash token IDs to get bytes)
    """
    
    def __init__(
        self,
        geometric_state: Dict[str, torch.Tensor],
        spectral_state: Dict[str, torch.Tensor],
        vocab_size: int = 4096,
        prime: int = 31,
    ):
        """Initialize with manifold state from both domains.
        
        Args:
            geometric_state: From manifold._step_geometric() or manifold.state
            spectral_state: From manifold._step_spectral() or manifold.modes
            vocab_size: Hash vocabulary size (for dehashing)
            prime: Hash prime (for dehashing)
        """
        self.geo = geometric_state
        self.spec = spectral_state
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask = vocab_size - 1
        
        # Precompute modular inverse for dehashing
        self.inv_prime = pow(prime, -1, vocab_size)
        
        # Extract key tensors
        self.positions = geometric_state.get("positions")
        self.velocities = geometric_state.get("velocities")
        self.energies = geometric_state.get("energies")
        self.heats = geometric_state.get("heats")
        self.excitations = geometric_state.get("excitations")  # = osc_omega
        self.token_ids = geometric_state.get("token_ids")
        self.masses = geometric_state.get("masses")
        
        self.mode_freqs = spectral_state.get("frequencies")
        self.mode_amps = spectral_state.get("amplitudes")
        self.mode_phases = spectral_state.get("phases")
        self.mode_conflict = spectral_state.get("conflict")
        self.mode_state = spectral_state.get("mode_state")
        self.mode_gate_widths = spectral_state.get("gate_widths")
        self.osc_phase = spectral_state.get("osc_phase")
        self.osc_energy = spectral_state.get("osc_energy")
        
        self.device = self.positions.device if self.positions is not None else torch.device("cpu")
        
        # Get number of active modes
        # Note: The num_modes tensor may be incorrect due to accumulation bugs
        # in the physics implementation. Use amplitude-based counting instead.
        if self.mode_amps is not None:
            # Count modes with non-trivial amplitude
            self.num_modes = int((self.mode_amps > 1e-6).sum().item())
        else:
            num_modes_tensor = spectral_state.get("num_modes")
            if num_modes_tensor is not None:
                if isinstance(num_modes_tensor, torch.Tensor):
                    self.num_modes = int(num_modes_tensor.item())
                else:
                    self.num_modes = int(num_modes_tensor)
            else:
                self.num_modes = 0
    
    # =========================================================================
    # Geometric Domain Queries
    # =========================================================================
    
    def hottest_particles(self, k: int = 10) -> GeometricQuery:
        """Find the k particles with highest energy."""
        if self.energies is None or self.energies.numel() == 0:
            return self._empty_geo_query()
        
        k = min(k, self.energies.numel())
        values, indices = torch.topk(self.energies, k)
        return self._geo_query_at(indices)
    
    def coldest_particles(self, k: int = 10) -> GeometricQuery:
        """Find the k particles with lowest energy (most 'dormant')."""
        if self.energies is None or self.energies.numel() == 0:
            return self._empty_geo_query()
        
        k = min(k, self.energies.numel())
        values, indices = torch.topk(self.energies, k, largest=False)
        return self._geo_query_at(indices)
    
    def most_heated_particles(self, k: int = 10) -> GeometricQuery:
        """Find the k particles with highest heat (entropy)."""
        if self.heats is None or self.heats.numel() == 0:
            return self._empty_geo_query()
        
        k = min(k, self.heats.numel())
        values, indices = torch.topk(self.heats, k)
        return self._geo_query_at(indices)
    
    def particles_near(self, position: torch.Tensor, radius: float) -> GeometricQuery:
        """Find particles within radius of a position."""
        if self.positions is None or self.positions.numel() == 0:
            return self._empty_geo_query()
        
        distances = torch.norm(self.positions - position.unsqueeze(0), dim=1)
        mask = distances < radius
        indices = torch.where(mask)[0]
        return self._geo_query_at(indices)
    
    def particles_with_token_id(self, token_id: int) -> GeometricQuery:
        """Find all particles with a specific token ID."""
        if self.token_ids is None or self.token_ids.numel() == 0:
            return self._empty_geo_query()
        
        mask = self.token_ids == token_id
        indices = torch.where(mask)[0]
        return self._geo_query_at(indices)
    
    def particles_in_range(self, start_idx: int, end_idx: int) -> GeometricQuery:
        """Get particles in an index range (useful for context/prompt)."""
        if self.positions is None:
            return self._empty_geo_query()
        
        end_idx = min(end_idx, self.positions.shape[0])
        indices = torch.arange(start_idx, end_idx, device=self.device)
        return self._geo_query_at(indices)
    
    def _geo_query_at(self, indices: torch.Tensor) -> GeometricQuery:
        """Build a GeometricQuery for given indices."""
        return GeometricQuery(
            indices=indices,
            positions=self.positions[indices] if self.positions is not None else torch.empty(0, 3, device=self.device),
            energies=self.energies[indices] if self.energies is not None else torch.empty(0, device=self.device),
            heats=self.heats[indices] if self.heats is not None else torch.empty(0, device=self.device),
            token_ids=self.token_ids[indices] if self.token_ids is not None else torch.empty(0, dtype=torch.int64, device=self.device),
            excitations=self.excitations[indices] if self.excitations is not None else torch.empty(0, device=self.device),
        )
    
    def _empty_geo_query(self) -> GeometricQuery:
        return GeometricQuery(
            indices=torch.empty(0, dtype=torch.int64, device=self.device),
            positions=torch.empty(0, 3, device=self.device),
            energies=torch.empty(0, device=self.device),
            heats=torch.empty(0, device=self.device),
            token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            excitations=torch.empty(0, device=self.device),
        )
    
    # =========================================================================
    # Coherence (ω-mode) Domain Queries
    # =========================================================================
    
    def crystallized_modes(self) -> ModeQuery:
        """Find all crystallized modes (state == 2)."""
        if self.mode_state is None or self.num_modes == 0:
            return self._empty_mode_query()
        
        mask = self.mode_state[:self.num_modes] == 2
        indices = torch.where(mask)[0]
        return self._mode_query_at(indices)
    
    def stable_modes(self) -> ModeQuery:
        """Find all stable modes (state >= 1)."""
        if self.mode_state is None or self.num_modes == 0:
            return self._empty_mode_query()
        
        mask = self.mode_state[:self.num_modes] >= 1
        indices = torch.where(mask)[0]
        return self._mode_query_at(indices)
    
    def most_coherent_modes(self, k: int = 5) -> ModeQuery:
        """Find k modes with lowest conflict (most coherent)."""
        if self.mode_conflict is None or self.num_modes == 0:
            return self._empty_mode_query()
        
        k = min(k, self.num_modes)
        values, indices = torch.topk(self.mode_conflict[:self.num_modes], k, largest=False)
        return self._mode_query_at(indices)
    
    def strongest_modes(self, k: int = 5) -> ModeQuery:
        """Find k modes with highest amplitude."""
        if self.mode_amps is None or self.num_modes == 0:
            return self._empty_mode_query()
        
        k = min(k, self.num_modes)
        values, indices = torch.topk(self.mode_amps[:self.num_modes], k)
        return self._mode_query_at(indices)
    
    def modes_at_frequency(self, freq: float, tolerance: float = 0.1) -> ModeQuery:
        """Find modes near a specific frequency."""
        if self.mode_freqs is None or self.num_modes == 0:
            return self._empty_mode_query()
        
        mask = torch.abs(self.mode_freqs[:self.num_modes] - freq) < tolerance
        indices = torch.where(mask)[0]
        return self._mode_query_at(indices)
    
    def _mode_query_at(self, indices: torch.Tensor) -> ModeQuery:
        """Build a ModeQuery for given indices."""
        return ModeQuery(
            mode_indices=indices,
            frequencies=self.mode_freqs[indices] if self.mode_freqs is not None else torch.empty(0, device=self.device),
            amplitudes=self.mode_amps[indices] if self.mode_amps is not None else torch.empty(0, device=self.device),
            phases=self.mode_phases[indices] if self.mode_phases is not None else torch.empty(0, device=self.device),
            conflict=self.mode_conflict[indices] if self.mode_conflict is not None else torch.empty(0, device=self.device),
            state=self.mode_state[indices] if self.mode_state is not None else torch.empty(0, dtype=torch.int32, device=self.device),
            gate_widths=self.mode_gate_widths[indices] if self.mode_gate_widths is not None else torch.empty(0, device=self.device),
        )
    
    def _empty_mode_query(self) -> ModeQuery:
        return ModeQuery(
            mode_indices=torch.empty(0, dtype=torch.int64, device=self.device),
            frequencies=torch.empty(0, device=self.device),
            amplitudes=torch.empty(0, device=self.device),
            phases=torch.empty(0, device=self.device),
            conflict=torch.empty(0, device=self.device),
            state=torch.empty(0, dtype=torch.int32, device=self.device),
            gate_widths=torch.empty(0, device=self.device),
        )
    
    # =========================================================================
    # Cross-Domain Queries (the key switching mechanism)
    # =========================================================================
    
    def modes_for_particles(self, geo_query: GeometricQuery, k: int = 3) -> ModeQuery:
        """Switch from geometric to coherence: find modes that couple to these particles.
        
        Uses the tuning kernel: T_ik = exp(-((ω_i - Ω_k)² / σ_k²))
        """
        if geo_query.excitations.numel() == 0 or self.num_modes == 0:
            return self._empty_mode_query()
        
        osc_omega = geo_query.excitations  # (M,)
        mode_omega = self.mode_freqs[:self.num_modes]  # (K,)
        mode_sigma = self.mode_gate_widths[:self.num_modes]  # (K,)
        
        # Compute coupling for each oscillator to each mode
        # Broadcasting: (M, 1) - (K,) -> (M, K)
        freq_diff = osc_omega.unsqueeze(1) - mode_omega.unsqueeze(0)
        sigma_sq = mode_sigma.unsqueeze(0) ** 2 + 1e-8
        coupling = torch.exp(-(freq_diff ** 2) / sigma_sq)  # (M, K)
        
        # Total coupling per mode (sum over oscillators)
        total_coupling = coupling.sum(dim=0)  # (K,)
        
        # Weight by mode amplitude and inverse conflict
        if self.mode_amps is not None and self.mode_conflict is not None:
            amp = self.mode_amps[:self.num_modes]
            conf = self.mode_conflict[:self.num_modes]
            score = total_coupling * amp * (1.0 - torch.clamp(conf, 0, 1))
        else:
            score = total_coupling
        
        # Top k modes
        k = min(k, self.num_modes)
        values, indices = torch.topk(score, k)
        return self._mode_query_at(indices)
    
    def particles_for_modes(self, mode_query: ModeQuery, k: int = 10) -> GeometricQuery:
        """Switch from coherence to geometric: find particles coupled to these modes.
        
        Returns the oscillators that resonate with the given modes.
        """
        if mode_query.frequencies.numel() == 0 or self.excitations is None:
            return self._empty_geo_query()
        
        mode_omega = mode_query.frequencies  # (C,)
        mode_sigma = mode_query.gate_widths  # (C,)
        osc_omega = self.excitations  # (N,)
        
        # Compute coupling for each oscillator to each mode
        # Broadcasting: (N, 1) - (C,) -> (N, C)
        freq_diff = osc_omega.unsqueeze(1) - mode_omega.unsqueeze(0)
        sigma_sq = mode_sigma.unsqueeze(0) ** 2 + 1e-8
        coupling = torch.exp(-(freq_diff ** 2) / sigma_sq)  # (N, C)
        
        # Total coupling per oscillator (sum over modes, weighted by amplitude)
        mode_amp = mode_query.amplitudes
        weighted_coupling = (coupling * mode_amp.unsqueeze(0)).sum(dim=1)  # (N,)
        
        # Top k oscillators
        k = min(k, weighted_coupling.numel())
        values, indices = torch.topk(weighted_coupling, k)
        return self._geo_query_at(indices)
    
    def coupling_strength(self, osc_idx: int, mode_idx: int) -> float:
        """Compute coupling strength between an oscillator and a mode."""
        if self.excitations is None or self.mode_freqs is None or self.mode_gate_widths is None:
            return 0.0
        
        osc_omega = float(self.excitations[osc_idx].item())
        mode_omega = float(self.mode_freqs[mode_idx].item())
        mode_sigma = float(self.mode_gate_widths[mode_idx].item())
        
        freq_diff = osc_omega - mode_omega
        coupling = np.exp(-(freq_diff ** 2) / (mode_sigma ** 2 + 1e-8))
        
        return coupling
    
    # =========================================================================
    # Inference Utilities
    # =========================================================================
    
    def score_candidate_bytes(
        self,
        context_indices: torch.Tensor,
        target_position: int,
        segment_size: Optional[int] = None,
    ) -> np.ndarray:
        """Score all 256 candidate bytes for the next position.
        
        Uses dual-domain inference:
        1. Find modes coupled to context oscillators (coherence)
        2. Score candidates by how well they couple to those modes
        
        Args:
            context_indices: Indices of context particles
            target_position: Position for the target byte
            segment_size: If set, position wraps every segment_size
        
        Returns:
            Array of 256 scores (higher = more likely)
        """
        scores = np.zeros(256, dtype=np.float32)
        
        # Step 1: Get context oscillators (geometric query)
        context_query = self._geo_query_at(context_indices)
        
        # Step 2: Find modes that couple to context (switch to coherence)
        coupled_modes = self.modes_for_particles(context_query, k=10)
        
        if coupled_modes.frequencies.numel() == 0:
            # Fallback: uniform scores
            return scores + 1.0 / 256
        
        # Step 3: Score each candidate byte by mode coupling
        pos = target_position
        if segment_size:
            pos = target_position % segment_size
        
        mode_omega = coupled_modes.frequencies.cpu().numpy()
        mode_sigma = coupled_modes.gate_widths.cpu().numpy()
        mode_amp = coupled_modes.amplitudes.cpu().numpy()
        mode_conf = coupled_modes.conflict.cpu().numpy()
        mode_state = coupled_modes.state.cpu().numpy()
        
        for byte_val in range(256):
            # Compute token ID and frequency for this candidate
            tid = (byte_val * self.prime + pos) & self.mask
            omega = tid * (2.0 / self.vocab_size)
            
            # Coupling to each mode
            freq_diff = omega - mode_omega
            coupling = np.exp(-(freq_diff ** 2) / (mode_sigma ** 2 + 1e-8))
            
            # Weight by amplitude, inverse conflict, and crystallization bonus
            weights = mode_amp * (1.0 - np.clip(mode_conf, 0, 1))
            # Bonus for crystallized modes
            weights *= np.where(mode_state == 2, 2.0, 1.0)
            
            scores[byte_val] = float(np.sum(coupling * weights))
        
        return scores
    
    def predict_next_byte(
        self,
        context_bytes: bytes,
        context_start_position: int,
        segment_size: Optional[int] = None,
        *,
        method: str = "wave",
        k_per_token: int = 64,
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """Predict the next byte from a byte-context.

        There are two supported methods:

        - "wave" (default): dual-domain "slide" inference. We map the context into
          the manifold (geometric particles), slide into wave space (modes), then
          score candidate next bytes by how strongly their would-be oscillator
          frequencies couple to the context-coupled modes.

          This is the intended post-refactor architecture: no explicit scanning
          for an exact matching substring in the training stream.

        - "ngram": legacy exact-context retrieval. We scan the training particle
          stream for a consecutive match of the hashed context, then vote using
          the immediately following token. This is useful as a baseline but can
          overstate generalization on templated datasets.
        
        The key insight: particles/oscillators ARE bytes. To predict:
        1. Find training particles that match the context pattern
        2. Look at what other particles are coupled to them (via modes or proximity)
        3. Among those, find particles at the "next" position
        4. Dehash those particles to get candidate bytes
        5. Vote by energy/coupling strength
        
        Args:
            context_bytes: The context bytes (e.g., last N bytes before target)
            context_start_position: Position of the first context byte
            segment_size: If set, positions wrap every segment_size
        
        Returns:
            (scores array of 256 values, list of (byte, score) top candidates)
        """
        if method not in ("wave", "ngram"):
            raise ValueError(f"Unknown method {method!r}; expected 'wave' or 'ngram'")

        if method == "wave":
            return self._predict_next_byte_wave(
                context_bytes=context_bytes,
                context_start_position=context_start_position,
                segment_size=segment_size,
                k_per_token=int(k_per_token),
            )
        # Legacy baseline.
        return self._predict_next_byte_ngram(
            context_bytes=context_bytes,
            context_start_position=context_start_position,
            segment_size=segment_size,
        )

    def _predict_next_byte_wave(
        self,
        *,
        context_bytes: bytes,
        context_start_position: int,
        segment_size: Optional[int],
        k_per_token: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """Wave-first next-byte prediction.

        Implementation sketch:
        - Compute hashed token IDs for each context byte at its (known) position.
        - For each context token ID, retrieve a small set of matching training
          particles (by energy) as anchors in geometric space.
        - Slide anchors -> modes (wave space) via mode coupling.
        - Score candidate bytes for the next position by coupling to those modes.
        """
        scores = np.zeros(256, dtype=np.float32)

        if self.token_ids is None or self.token_ids.numel() == 0:
            return scores + 1.0 / 256, []

        token_ids = self.token_ids.detach().to("cpu").to(torch.int64).numpy()
        energies = (
            self.energies.detach().to("cpu").to(torch.float32).numpy()
            if self.energies is not None and self.energies.numel() == self.token_ids.numel()
            else np.ones(int(token_ids.shape[0]), dtype=np.float32)
        )

        # Target (next) position in the same positional frame as hashing.
        target_position = int(context_start_position) + int(len(context_bytes))

        # Anchor selection: for each context byte, gather up to k_per_token
        # training particles with matching token ID.
        anchor_indices: list[int] = []
        ctx_len = int(len(context_bytes))
        for i in range(ctx_len):
            byte_val = int(context_bytes[i])
            pos = int(context_start_position) + i
            if segment_size is not None:
                pos = int(pos % int(segment_size))
            tid = int((byte_val * self.prime + pos) & self.mask)

            idxs = np.nonzero(token_ids == tid)[0]
            if idxs.size == 0:
                continue

            k = int(k_per_token)
            if k > 0 and idxs.size > k:
                # Prefer high-energy anchors (more "active" carriers).
                # argsort on a small slice: O(m log m) where m = idxs.size.
                o = np.argsort(energies[idxs])[::-1][:k]
                idxs = idxs[o]

            anchor_indices.extend(int(x) for x in idxs.tolist())

        if not anchor_indices:
            return scores + 1.0 / 256, []

        # Unique anchors to cap coupling compute.
        anchor_indices = list(sorted(set(anchor_indices)))
        context_indices = torch.tensor(anchor_indices, device=self.device, dtype=torch.int64)

        scores = self.score_candidate_bytes(
            context_indices=context_indices,
            target_position=int(target_position),
            segment_size=segment_size,
        ).astype(np.float32)

        # score_candidate_bytes returns unnormalized scores; normalize to a distribution.
        s = float(scores.sum())
        if s > 0.0 and np.isfinite(s):
            scores = scores / s
        else:
            scores = np.ones(256, dtype=np.float32) / 256

        top_indices = np.argsort(scores)[::-1][:10]
        top_candidates = [(int(idx), float(scores[idx])) for idx in top_indices]
        return scores, top_candidates

    def _predict_next_byte_ngram(
        self,
        *,
        context_bytes: bytes,
        context_start_position: int,
        segment_size: Optional[int],
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """Legacy exact-context retrieval baseline (see `predict_next_byte`)."""
        scores = np.zeros(256, dtype=np.float32)
        
        if self.token_ids is None or self.token_ids.numel() == 0:
            return scores + 1.0 / 256, []
        
        token_ids = self.token_ids.cpu().numpy()
        energies = self.energies.cpu().numpy() if self.energies is not None else np.ones(len(token_ids))
        n_particles = len(token_ids)
        
        # Step 1: Compute token IDs for context bytes
        context_tids = []
        for i, byte_val in enumerate(context_bytes):
            pos = context_start_position + i
            if segment_size:
                pos = pos % segment_size
            tid = (byte_val * self.prime + pos) & self.mask
            context_tids.append(tid)
        
        # Step 2: Find training particles that match the context pattern
        # Look for sequences where these token IDs appear consecutively
        context_len = len(context_tids)
        match_end_positions = []  # Positions where context pattern ends
        
        for start_idx in range(n_particles - context_len):
            # Check if tokens at [start_idx : start_idx + context_len] match context_tids
            match = True
            for j, ctx_tid in enumerate(context_tids):
                if token_ids[start_idx + j] != ctx_tid:
                    match = False
                    break
            if match:
                # The context ends at start_idx + context_len - 1
                # The "next" particle would be at start_idx + context_len
                next_idx = start_idx + context_len
                if next_idx < n_particles:
                    match_end_positions.append(next_idx)
        
        # Step 3: For each matching position, get the "next" particle and dehash it
        if match_end_positions:
            for next_idx in match_end_positions:
                next_tid = int(token_ids[next_idx])
                next_energy = float(energies[next_idx])
                
                # Dehash: what position was this particle at?
                # We need to know the position to dehash properly
                next_pos = next_idx  # Absolute position in sequence
                if segment_size:
                    next_pos = next_idx % segment_size
                
                # Dehash to get byte
                byte_val = self.dehash_token_id(next_tid, next_pos)
                if 0 <= byte_val < 256:
                    scores[byte_val] += next_energy
        
        # Step 4: Also look at mode-coupled particles
        # Find particles that are coupled via modes to the context
        if self.num_modes > 0 and len(match_end_positions) > 0:
            # Get context particle indices (where context matched)
            context_particle_indices = []
            for end_pos in match_end_positions:
                start_pos = end_pos - context_len
                context_particle_indices.extend(range(start_pos, end_pos))
            
            if context_particle_indices:
                context_indices = torch.tensor(
                    list(set(context_particle_indices)), 
                    device=self.device, 
                    dtype=torch.int64
                )
                
                # Find modes coupled to context
                context_query = self._geo_query_at(context_indices)
                coupled_modes = self.modes_for_particles(context_query, k=5)
                
                if coupled_modes.frequencies.numel() > 0:
                    # Find particles coupled to those modes
                    coupled_particles = self.particles_for_modes(coupled_modes, k=20)
                    
                    # Check which of these are at "next" positions
                    for idx_tensor in [coupled_particles.indices]:
                        for idx in idx_tensor.cpu().numpy():
                            # Is this particle at a position right after a context match?
                            for end_pos in match_end_positions:
                                if idx == end_pos:  # It's the "next" particle
                                    tid = int(token_ids[idx])
                                    pos = idx if segment_size is None else idx % segment_size
                                    byte_val = self.dehash_token_id(tid, pos)
                                    if 0 <= byte_val < 256:
                                        energy = float(energies[idx])
                                        # Add bonus for mode coupling
                                        scores[byte_val] += energy * 0.5
        
        # Normalize scores
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones(256, dtype=np.float32) / 256
        
        # Get top candidates
        top_indices = np.argsort(scores)[::-1][:10]
        top_candidates = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return scores, top_candidates
    
    def dehash_token_id(self, token_id: int, position: int) -> int:
        """Reverse the hash to get the original byte value.
        
        Hash: token_id = (byte * prime + position) & mask
        Inverse: byte = ((token_id - position) * inv_prime) & mask
        """
        target = (token_id - position) & self.mask
        byte_val = (target * self.inv_prime) & self.mask
        return byte_val if byte_val < 256 else -1
    
    def dehash_particles(self, geo_query: GeometricQuery, positions: torch.Tensor) -> bytes:
        """Dehash a set of particles back to bytes.
        
        Args:
            geo_query: Particles to dehash
            positions: Sequence positions for each particle
        
        Returns:
            Reconstructed bytes
        """
        result = []
        token_ids = geo_query.token_ids.cpu().numpy()
        pos_np = positions.cpu().numpy() if isinstance(positions, torch.Tensor) else positions
        
        for tid, pos in zip(token_ids, pos_np):
            byte_val = self.dehash_token_id(int(tid), int(pos))
            if 0 <= byte_val < 256:
                result.append(byte_val)
            else:
                result.append(0)  # Invalid, use null byte
        
        return bytes(result)
