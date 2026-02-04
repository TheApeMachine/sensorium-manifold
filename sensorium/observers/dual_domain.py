"""Dual-domain inference utilities.

The Sensorium Manifold has two observation domains:
1. Geometric: particles with positions, velocities, energies, heats
2. Spectral: oscillators coupled to carriers via resonance

Inference involves switching between domains to find answers:
- Geometric: for locality, spatial clustering, energy "hotness"
- Spectral: for resonance, phase coherence, carrier coupling

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
class SpectralQuery:
    """Query result from the spectral domain."""
    carrier_indices: torch.Tensor   # Carrier indices
    frequencies: torch.Tensor       # Carrier ω
    amplitudes: torch.Tensor        # Carrier |C|
    phases: torch.Tensor            # Carrier ψ
    conflict: torch.Tensor          # Carrier conflict (lower = more coherent)
    state: torch.Tensor             # 0=volatile, 1=stable, 2=crystallized
    gate_widths: torch.Tensor       # Carrier σ (frequency selectivity)


class DualDomainInference:
    """Inference by switching between geometric and spectral domains.
    
    Usage pattern:
    1. Start in one domain (e.g., find "hot" particles)
    2. Switch to other domain (e.g., find carriers they couple to)
    3. Switch back (e.g., find other particles coupled to those carriers)
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
            spectral_state: From manifold._step_spectral() or manifold.carriers
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
        
        self.carrier_freqs = spectral_state.get("frequencies")
        self.carrier_amps = spectral_state.get("amplitudes")
        self.carrier_phases = spectral_state.get("phases")
        self.carrier_conflict = spectral_state.get("conflict")
        self.carrier_state = spectral_state.get("carrier_state")
        self.carrier_gate_widths = spectral_state.get("gate_widths")
        self.osc_phase = spectral_state.get("osc_phase")
        self.osc_energy = spectral_state.get("osc_energy")
        
        self.device = self.positions.device if self.positions is not None else torch.device("cpu")
        
        # Get number of active carriers
        num_carriers_tensor = spectral_state.get("num_carriers")
        if num_carriers_tensor is not None:
            if isinstance(num_carriers_tensor, torch.Tensor):
                self.num_carriers = int(num_carriers_tensor.item())
            else:
                self.num_carriers = int(num_carriers_tensor)
        else:
            # Fallback: count non-zero amplitudes
            if self.carrier_amps is not None:
                self.num_carriers = int((self.carrier_amps > 1e-6).sum().item())
            else:
                self.num_carriers = 0
    
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
    # Spectral Domain Queries
    # =========================================================================
    
    def crystallized_carriers(self) -> SpectralQuery:
        """Find all crystallized carriers (state == 2)."""
        if self.carrier_state is None or self.num_carriers == 0:
            return self._empty_spec_query()
        
        mask = self.carrier_state[:self.num_carriers] == 2
        indices = torch.where(mask)[0]
        return self._spec_query_at(indices)
    
    def stable_carriers(self) -> SpectralQuery:
        """Find all stable carriers (state >= 1)."""
        if self.carrier_state is None or self.num_carriers == 0:
            return self._empty_spec_query()
        
        mask = self.carrier_state[:self.num_carriers] >= 1
        indices = torch.where(mask)[0]
        return self._spec_query_at(indices)
    
    def most_coherent_carriers(self, k: int = 5) -> SpectralQuery:
        """Find k carriers with lowest conflict (most coherent)."""
        if self.carrier_conflict is None or self.num_carriers == 0:
            return self._empty_spec_query()
        
        k = min(k, self.num_carriers)
        # Lower conflict = more coherent
        values, indices = torch.topk(self.carrier_conflict[:self.num_carriers], k, largest=False)
        return self._spec_query_at(indices)
    
    def strongest_carriers(self, k: int = 5) -> SpectralQuery:
        """Find k carriers with highest amplitude."""
        if self.carrier_amps is None or self.num_carriers == 0:
            return self._empty_spec_query()
        
        k = min(k, self.num_carriers)
        values, indices = torch.topk(self.carrier_amps[:self.num_carriers], k)
        return self._spec_query_at(indices)
    
    def carriers_at_frequency(self, freq: float, tolerance: float = 0.1) -> SpectralQuery:
        """Find carriers near a specific frequency."""
        if self.carrier_freqs is None or self.num_carriers == 0:
            return self._empty_spec_query()
        
        mask = torch.abs(self.carrier_freqs[:self.num_carriers] - freq) < tolerance
        indices = torch.where(mask)[0]
        return self._spec_query_at(indices)
    
    def _spec_query_at(self, indices: torch.Tensor) -> SpectralQuery:
        """Build a SpectralQuery for given indices."""
        return SpectralQuery(
            carrier_indices=indices,
            frequencies=self.carrier_freqs[indices] if self.carrier_freqs is not None else torch.empty(0, device=self.device),
            amplitudes=self.carrier_amps[indices] if self.carrier_amps is not None else torch.empty(0, device=self.device),
            phases=self.carrier_phases[indices] if self.carrier_phases is not None else torch.empty(0, device=self.device),
            conflict=self.carrier_conflict[indices] if self.carrier_conflict is not None else torch.empty(0, device=self.device),
            state=self.carrier_state[indices] if self.carrier_state is not None else torch.empty(0, dtype=torch.int32, device=self.device),
            gate_widths=self.carrier_gate_widths[indices] if self.carrier_gate_widths is not None else torch.empty(0, device=self.device),
        )
    
    def _empty_spec_query(self) -> SpectralQuery:
        return SpectralQuery(
            carrier_indices=torch.empty(0, dtype=torch.int64, device=self.device),
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
    
    def carriers_for_particles(self, geo_query: GeometricQuery, k: int = 3) -> SpectralQuery:
        """Switch from geometric to spectral: find carriers that couple to these particles.
        
        Uses the tuning kernel: T_ik = exp(-((ω_i - Ω_k)² / σ_k²))
        """
        if geo_query.excitations.numel() == 0 or self.num_carriers == 0:
            return self._empty_spec_query()
        
        osc_omega = geo_query.excitations  # (M,)
        carrier_omega = self.carrier_freqs[:self.num_carriers]  # (K,)
        carrier_sigma = self.carrier_gate_widths[:self.num_carriers]  # (K,)
        
        # Compute coupling for each oscillator to each carrier
        # Broadcasting: (M, 1) - (K,) -> (M, K)
        freq_diff = osc_omega.unsqueeze(1) - carrier_omega.unsqueeze(0)
        sigma_sq = carrier_sigma.unsqueeze(0) ** 2 + 1e-8
        coupling = torch.exp(-(freq_diff ** 2) / sigma_sq)  # (M, K)
        
        # Total coupling per carrier (sum over oscillators)
        total_coupling = coupling.sum(dim=0)  # (K,)
        
        # Weight by carrier amplitude and inverse conflict
        if self.carrier_amps is not None and self.carrier_conflict is not None:
            amp = self.carrier_amps[:self.num_carriers]
            conf = self.carrier_conflict[:self.num_carriers]
            score = total_coupling * amp * (1.0 - torch.clamp(conf, 0, 1))
        else:
            score = total_coupling
        
        # Top k carriers
        k = min(k, self.num_carriers)
        values, indices = torch.topk(score, k)
        return self._spec_query_at(indices)
    
    def particles_for_carriers(self, spec_query: SpectralQuery, k: int = 10) -> GeometricQuery:
        """Switch from spectral to geometric: find particles coupled to these carriers.
        
        Returns the oscillators that resonate with the given carriers.
        """
        if spec_query.frequencies.numel() == 0 or self.excitations is None:
            return self._empty_geo_query()
        
        carrier_omega = spec_query.frequencies  # (C,)
        carrier_sigma = spec_query.gate_widths  # (C,)
        osc_omega = self.excitations  # (N,)
        
        # Compute coupling for each oscillator to each carrier
        # Broadcasting: (N, 1) - (C,) -> (N, C)
        freq_diff = osc_omega.unsqueeze(1) - carrier_omega.unsqueeze(0)
        sigma_sq = carrier_sigma.unsqueeze(0) ** 2 + 1e-8
        coupling = torch.exp(-(freq_diff ** 2) / sigma_sq)  # (N, C)
        
        # Total coupling per oscillator (sum over carriers, weighted by amplitude)
        carrier_amp = spec_query.amplitudes
        weighted_coupling = (coupling * carrier_amp.unsqueeze(0)).sum(dim=1)  # (N,)
        
        # Top k oscillators
        k = min(k, weighted_coupling.numel())
        values, indices = torch.topk(weighted_coupling, k)
        return self._geo_query_at(indices)
    
    def coupling_strength(self, osc_idx: int, carrier_idx: int) -> float:
        """Compute coupling strength between an oscillator and a carrier."""
        if self.excitations is None or self.carrier_freqs is None:
            return 0.0
        
        osc_omega = float(self.excitations[osc_idx].item())
        carrier_omega = float(self.carrier_freqs[carrier_idx].item())
        carrier_sigma = float(self.carrier_gate_widths[carrier_idx].item())
        
        freq_diff = osc_omega - carrier_omega
        coupling = np.exp(-(freq_diff ** 2) / (carrier_sigma ** 2 + 1e-8))
        
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
        1. Find carriers coupled to context oscillators (spectral)
        2. Score candidates by how well they couple to those carriers
        
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
        
        # Step 2: Find carriers that couple to context (switch to spectral)
        coupled_carriers = self.carriers_for_particles(context_query, k=10)
        
        if coupled_carriers.frequencies.numel() == 0:
            # Fallback: uniform scores
            return scores + 1.0 / 256
        
        # Step 3: Score each candidate byte by carrier coupling
        pos = target_position
        if segment_size:
            pos = target_position % segment_size
        
        carrier_omega = coupled_carriers.frequencies.cpu().numpy()
        carrier_sigma = coupled_carriers.gate_widths.cpu().numpy()
        carrier_amp = coupled_carriers.amplitudes.cpu().numpy()
        carrier_conf = coupled_carriers.conflict.cpu().numpy()
        carrier_state = coupled_carriers.state.cpu().numpy()
        
        for byte_val in range(256):
            # Compute token ID and frequency for this candidate
            tid = (byte_val * self.prime + pos) & self.mask
            omega = tid * (2.0 / self.vocab_size)
            
            # Coupling to each carrier
            freq_diff = omega - carrier_omega
            coupling = np.exp(-(freq_diff ** 2) / (carrier_sigma ** 2 + 1e-8))
            
            # Weight by amplitude, inverse conflict, and crystallization bonus
            weights = carrier_amp * (1.0 - np.clip(carrier_conf, 0, 1))
            # Bonus for crystallized carriers
            weights *= np.where(carrier_state == 2, 2.0, 1.0)
            
            scores[byte_val] = float(np.sum(coupling * weights))
        
        return scores
    
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
