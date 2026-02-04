from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .gpu_radix_trie import GpuRadixTrie

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

        # GPU-resident radix trie lookup structure (rebuilt when sequences change).
        # Pad token is vocab_size, which is outside the normal token range [0, vocab_size-1].
        self._pad_id = int(self.vocab_size)
        self.seq_pad = torch.empty(0, self.order, device=device, dtype=torch.long)
        self._trie_gpu = GpuRadixTrie(device=device, pad_id=self._pad_id)
        self._trie_dirty = True

        # Baseline for binding energies (homeostatic scale), tracked per sequence length.
        self._binding_baseline_by_len: Dict[int, torch.Tensor] = {}

    @property
    def num_chunks(self) -> int:
        return int(self.energy.numel())

    def _pad_seq(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("seq must have shape [N,L]")
        if int(seq.shape[1]) > self.order:
            raise ValueError("seq length exceeds maximum order")
        n = int(seq.shape[0])
        pad = torch.full((n, self.order), self._pad_id, device=self.device, dtype=torch.long)
        pad[:, : int(seq.shape[1])] = seq
        return pad

    def _rebuild_trie(self) -> None:
        if self.num_chunks == 0:
            self._trie_gpu.clear()
            self._trie_dirty = False
            return
        values = torch.arange(self.num_chunks, device=self.device, dtype=torch.long)
        self._trie_gpu.rebuild(self.seq_pad, values)
        self._trie_dirty = False

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

        if self.num_chunks == 0:
            key = torch.zeros((n,), device=self.device, dtype=torch.long)
            return LookupResult(idx=idx, exists=exists, key=key)

        if self._trie_dirty:
            self._rebuild_trie()

        idx, exists = self._trie_gpu.lookup(seq)

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

        # Coalesce duplicates on GPU.
        seq_u, inv = torch.unique(seq_p, dim=0, return_inverse=True)
        mass_u = torch.zeros(int(seq_u.shape[0]), device=self.device, dtype=torch.float32)
        mass_u.scatter_add_(0, inv, mass_p)

        lookup_u = self.lookup(seq_u)
        idx_existing = lookup_u.idx
        exists = lookup_u.exists

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

            # Update GPU sequence table (used for trie rebuilds)
            seq_add_pad = self._pad_seq(seq_add)
            self.seq_pad = torch.cat([self.seq_pad, seq_add_pad], dim=0)
            self._trie_dirty = True

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
