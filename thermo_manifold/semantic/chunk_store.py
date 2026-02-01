from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .radix_trie import RadixTrie


@dataclass
class LookupResult:
    idx: torch.Tensor
    exists: torch.Tensor
    key: torch.Tensor  # retained for compatibility; 0 for variable-length mode


class ChunkStore:
    """Dynamic store of higher-order "slow" particles (chunks).

    Key design constraint: chunk indices must remain stable once assigned, because
    other structures (e.g., chunk->token bond graphs) reference them.

    We therefore maintain:
    - An append-only set of chunk state tensors indexed by chunk_id (0..C-1)
    - A separate sorted key->chunk_id mapping for fast GPU lookups.

    Chunk sequences are variable-length; we maintain a radix trie for lookup.
    """

    def __init__(
        self,
        *,
        order: int,
        vocab_size: int,
        sem_dim: int,
        device: torch.device,
        eps: float,
    ):
        if order < 2:
            raise ValueError('order must be >= 2')
        # `order` is treated as a maximum expected order (used only for validation of inputs
        # if callers still pass fixed-length sequences). Actual stored chunks can vary in length.
        self.order = int(order)
        self.vocab_size = int(vocab_size)
        self.sem_dim = int(sem_dim)
        self.device = device
        self.eps = float(eps)

        # Stable chunk state (append-only)
        # Variable-length sequences are stored in a flat buffer with offsets.
        self.seq_flat = torch.empty(0, device=device, dtype=torch.long)
        self.seq_offsets = torch.zeros(1, device=device, dtype=torch.long)  # len = num_chunks+1
        self.seq_len = torch.empty(0, device=device, dtype=torch.long)
        self.position = torch.empty(0, self.sem_dim, device=device, dtype=torch.float32)
        self.energy = torch.empty(0, device=device, dtype=torch.float32)
        self.excitation = torch.empty(0, device=device, dtype=torch.float32)
        self.heat = torch.empty(0, device=device, dtype=torch.float32)

        # Radix trie mapping sequence tuples -> chunk_id.
        self._trie = RadixTrie()

        # Baseline for binding energies (homeostatic scale), tracked per sequence length.
        self._binding_baseline_by_len: Dict[int, torch.Tensor] = {}

    @property
    def num_chunks(self) -> int:
        return int(self.energy.numel())

    # ----------------------------
    # Lookup
    # ----------------------------

    def lookup(self, seq: torch.Tensor) -> LookupResult:
        """Lookup chunk_id(s) for a batch of sequences."""
        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim == 1:
            seq = seq.view(1, -1)
        if seq.ndim != 2 or seq.shape[1] < 2:
            raise ValueError("seq must have shape [N,L] with L>=2")

        n = int(seq.shape[0])
        idx = torch.full((n,), -1, dtype=torch.long, device=self.device)
        exists = torch.zeros((n,), dtype=torch.bool, device=self.device)

        # Trie operates on CPU tuples; overhead is small because lookups are locality-bound.
        seq_cpu = seq.detach().to("cpu")
        for i in range(n):
            key_t = tuple(int(x) for x in seq_cpu[i].tolist())
            hit = self._trie.get(key_t)
            if hit is not None:
                idx[i] = int(hit)
                exists[i] = True

        key = torch.zeros((n,), device=self.device, dtype=torch.long)
        return LookupResult(idx=idx, exists=exists, key=key)

    # ----------------------------
    # Homeostatic baseline
    # ----------------------------

    def update_binding_baseline(self, binding: torch.Tensor, *, length: int, dt: float) -> torch.Tensor:
        """Update and return the binding baseline for a given length (scalar tensor)."""

        eps = self.eps
        b = binding.to(device=self.device, dtype=torch.float32).mean()
        ln = int(length)
        base = self._binding_baseline_by_len.get(ln)
        if base is None:
            base = b.detach().clone()
            self._binding_baseline_by_len[ln] = base
            return base
        alpha = float(dt) / (float(dt) + float(base.abs().item()) + eps)
        base_new = base * (1.0 - alpha) + b.detach() * alpha
        self._binding_baseline_by_len[ln] = base_new
        return base_new

    def binding_baseline(self, length: int) -> torch.Tensor:
        base = self._binding_baseline_by_len.get(int(length))
        if base is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return base

    # ----------------------------
    # Insertion / reinforcement
    # ----------------------------

    def add_or_reinforce(self, seq: torch.Tensor, mass: torch.Tensor, *, word_pos: torch.Tensor) -> LookupResult:
        """Add new chunks or reinforce existing ones.

        seq: [N,L] where L>=2 (variable-length supported)
        mass: scalar or [N]
        word_pos: [V,D] token embeddings (used to place new chunks)

        Returns a LookupResult aligned with the input seq rows.
        """

        eps = self.eps

        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim == 1:
            seq = seq.view(1, -1)
        if seq.ndim != 2 or seq.shape[1] < 2:
            raise ValueError("seq must have shape [N,L] with L>=2")

        mass = mass.to(device=self.device, dtype=torch.float32)
        if mass.numel() == 1:
            mass = mass.expand(int(seq.shape[0]))
        else:
            mass = mass.flatten()
            if mass.shape[0] != seq.shape[0]:
                raise ValueError('mass must be scalar or have shape [N]')

        # Filter: only positive mass contributes to structural reinforcement.
        pos_mask = mass > 0
        if not bool(pos_mask.any()):
            return self.lookup(seq)

        seq_p = seq[pos_mask]
        mass_p = mass[pos_mask]

        # Coalesce duplicates on CPU (locality-bound).
        seq_cpu = seq_p.detach().to("cpu")
        mass_cpu = mass_p.detach().to("cpu")
        accum: Dict[Tuple[int, ...], float] = {}
        reps: Dict[Tuple[int, ...], torch.Tensor] = {}
        for i in range(int(seq_cpu.shape[0])):
            key_t = tuple(int(x) for x in seq_cpu[i].tolist())
            accum[key_t] = accum.get(key_t, 0.0) + float(mass_cpu[i].item())
            if key_t not in reps:
                reps[key_t] = seq_p[i]

        keys = list(accum.keys())
        seq_u = torch.stack([reps[k] for k in keys], dim=0)
        mass_u = torch.tensor([accum[k] for k in keys], device=self.device, dtype=torch.float32)

        idx_existing = torch.full((int(seq_u.shape[0]),), -1, device=self.device, dtype=torch.long)
        exists = torch.zeros((int(seq_u.shape[0]),), device=self.device, dtype=torch.bool)
        for i, k in enumerate(keys):
            hit = self._trie.get(k)
            if hit is not None:
                idx_existing[i] = int(hit)
                exists[i] = True

        # Reinforce existing chunks.
        if bool(exists.any()):
            idx_e = idx_existing[exists]
            m_e = mass_u[exists]
            self.energy[idx_e] = self.energy[idx_e] + m_e
            self.excitation[idx_e] = self.excitation[idx_e] + m_e

        # Insert new chunks.
        add_mask = ~exists
        if bool(add_mask.any()):
            seq_add = seq_u[add_mask]
            mass_add = mass_u[add_mask]

            # Append stable chunk state.
            start = self.num_chunks
            add_n = int(seq_add.shape[0])
            idx_add = torch.arange(start, start + add_n, device=self.device, dtype=torch.long)

            # Place new chunks at the normalized sum of constituent token embeddings.
            wp = word_pos.to(device=self.device, dtype=torch.float32)
            pos_add = wp[seq_add].sum(dim=1)
            pos_add = pos_add / (pos_add.norm(dim=1, keepdim=True) + eps)

            self.position = torch.cat([self.position, pos_add], dim=0)
            self.energy = torch.cat([self.energy, mass_add], dim=0)
            self.excitation = torch.cat([self.excitation, mass_add], dim=0)
            self.heat = torch.cat([self.heat, torch.zeros(add_n, device=self.device, dtype=torch.float32)], dim=0)

            # Append sequences to flat buffer + offsets.
            flat_add = seq_add.reshape(-1)
            self.seq_flat = torch.cat([self.seq_flat, flat_add], dim=0)
            lens_add = torch.full((add_n,), int(seq_add.shape[1]), device=self.device, dtype=torch.long)
            self.seq_len = torch.cat([self.seq_len, lens_add], dim=0)
            last = int(self.seq_offsets[-1].item())
            new_offsets = last + torch.cumsum(lens_add, dim=0)
            self.seq_offsets = torch.cat([self.seq_offsets, new_offsets.to(self.seq_offsets.dtype)], dim=0)

            # Update trie
            seq_add_cpu = seq_add.detach().to("cpu")
            for i in range(add_n):
                key_t = tuple(int(x) for x in seq_add_cpu[i].tolist())
                self._trie.insert(key_t, int(idx_add[i].item()))

        return self.lookup(seq)

    # ----------------------------
    # Dynamics
    # ----------------------------

    def decay(self, *, ratio: torch.Tensor, dt: float) -> None:
        """Homeostatic decay for chunk reservoirs (no reinforcement)."""

        if self.num_chunks == 0:
            return

        eps = self.eps
        dt = float(dt)
        ratio = ratio.to(device=self.device, dtype=torch.float32)

        e = self.energy
        x = self.excitation
        h = self.heat

        e_scale = e.abs().mean() + eps
        x_scale = x.abs().mean() + eps
        h_scale = h.abs().mean() + eps

        e_decay = torch.exp(-dt * ratio / e_scale)
        x_decay = torch.exp(-dt * ratio / x_scale)
        h_decay = torch.exp(-dt * ratio / h_scale)

        self.energy = (e * e_decay).clamp(min=0.0)
        self.excitation = (x * x_decay).clamp(min=0.0)
        self.heat = (h * h_decay).clamp(min=0.0)

    def distribution(self) -> torch.Tensor:
        """Return a normalized distribution over chunks based on excitation."""

        if self.num_chunks == 0:
            return torch.zeros(0, device=self.device, dtype=torch.float32)

        eps = self.eps
        x = self.excitation.clamp(min=0.0)
        s = x.sum()
        if float(s.item()) <= eps:
            return torch.zeros_like(x)
        return x / (s + eps)
