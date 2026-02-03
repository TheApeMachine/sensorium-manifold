from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _isin_sorted(query: torch.Tensor, sorted_set: torch.Tensor) -> torch.Tensor:
    """Vectorized membership test: query in sorted_set.

    Both tensors must be 1D on the same device.
    """

    if query.numel() == 0:
        return torch.zeros(0, device=query.device, dtype=torch.bool)
    if sorted_set.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)

    idx = torch.searchsorted(sorted_set, query)
    in_range = idx < sorted_set.numel()
    idx_safe = idx.clamp(max=max(int(sorted_set.numel()) - 1, 0))
    hit = sorted_set[idx_safe] == query
    return in_range & hit


@dataclass
class BondBatch:
    eidx: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    w: torch.Tensor
    trace: torch.Tensor


class SparseBondGraph:
    """GPU-friendly sparse directed bond graph.

    Goals:
    - No Python dicts or per-edge loops.
    - Dynamic insertion via sorted-key merge (no full re-sort of the whole graph).
    - Edge-local state: mass (w) and eligibility trace.
    """

    def __init__(self, num_nodes: int, *, device: torch.device, dtype: torch.dtype, eps: float):
        self.num_nodes = int(num_nodes)
        self.device = device
        self.dtype = dtype
        self.eps = float(eps)

        # Sorted by key = src*num_nodes + dst.
        self.key = torch.empty(0, device=device, dtype=torch.long)
        self.src = torch.empty(0, device=device, dtype=torch.long)
        self.dst = torch.empty(0, device=device, dtype=torch.long)
        self.w = torch.empty(0, device=device, dtype=dtype)
        self.trace = torch.empty(0, device=device, dtype=dtype)

    @property
    def num_edges(self) -> int:
        return int(self.key.numel())

    # ----------------------------
    # Insertion (vectorized)
    # ----------------------------

    def add_edges(self, src: torch.Tensor, dst: torch.Tensor, mass: torch.Tensor) -> None:
        """Add or reinforce edges.

        src, dst: [E]
        mass: scalar or [E]
        """

        src = src.to(device=self.device, dtype=torch.long).flatten()
        dst = dst.to(device=self.device, dtype=torch.long).flatten()
        if src.numel() == 0:
            return

        if src.shape != dst.shape:
            raise ValueError("src and dst must have the same shape")

        mass = mass.to(device=self.device, dtype=self.dtype)
        if mass.numel() == 1:
            mass = mass.expand_as(src).contiguous()
        else:
            mass = mass.flatten()
            if mass.shape != src.shape:
                raise ValueError("mass must be scalar or have the same shape as src/dst")

        # Clamp node IDs into range.
        src = src.clamp(min=0, max=self.num_nodes - 1)
        dst = dst.clamp(min=0, max=self.num_nodes - 1)

        key_new = src * self.num_nodes + dst

        # Coalesce duplicates in the *new* batch (typically small).
        order = torch.argsort(key_new)
        key_new = key_new[order]
        src = src[order]
        dst = dst[order]
        mass = mass[order]

        key_u, inv = torch.unique_consecutive(key_new, return_inverse=True)
        if key_u.numel() != key_new.numel():
            mass_u = torch.zeros(int(key_u.numel()), device=self.device, dtype=self.dtype)
            mass_u.index_add_(0, inv, mass)
            # src/dst can be reconstructed from key_u.
            src_u = (key_u // self.num_nodes).to(torch.long)
            dst_u = (key_u % self.num_nodes).to(torch.long)
        else:
            mass_u = mass
            src_u = src
            dst_u = dst

        # New edges start with trace equal to their injected mass (eligibility from observation).
        trace_u = mass_u.clone()

        if self.num_edges == 0:
            self.key = key_u
            self.src = src_u
            self.dst = dst_u
            self.w = mass_u
            self.trace = trace_u
            return

        # Locate keys in existing sorted key vector.
        pos = torch.searchsorted(self.key, key_u)
        exists = (pos < self.key.numel()) & (self.key[pos.clamp(max=self.key.numel() - 1)] == key_u)

        # Reinforce existing.
        if exists.any():
            p = pos[exists]
            self.w[p] = self.w[p] + mass_u[exists]
            self.trace[p] = self.trace[p] + mass_u[exists]

        # Insert novel edges.
        add_mask = ~exists
        if not add_mask.any():
            return

        key_add = key_u[add_mask]
        src_add = src_u[add_mask]
        dst_add = dst_u[add_mask]
        w_add = mass_u[add_mask]
        tr_add = trace_u[add_mask]
        pos_add = pos[add_mask]

        # Merge two sorted arrays without a full sort.
        old_n = int(self.num_edges)
        add_n = int(key_add.numel())
        merged_n = old_n + add_n

        new_pos = pos_add + torch.arange(add_n, device=self.device, dtype=torch.long)
        old_i = torch.arange(old_n, device=self.device, dtype=torch.long)
        shift = torch.searchsorted(pos_add, old_i, right=True)
        old_pos = old_i + shift

        key_m = torch.empty(merged_n, device=self.device, dtype=torch.long)
        src_m = torch.empty(merged_n, device=self.device, dtype=torch.long)
        dst_m = torch.empty(merged_n, device=self.device, dtype=torch.long)
        w_m = torch.empty(merged_n, device=self.device, dtype=self.dtype)
        tr_m = torch.empty(merged_n, device=self.device, dtype=self.dtype)

        key_m[old_pos] = self.key
        src_m[old_pos] = self.src
        dst_m[old_pos] = self.dst
        w_m[old_pos] = self.w
        tr_m[old_pos] = self.trace

        key_m[new_pos] = key_add
        src_m[new_pos] = src_add
        dst_m[new_pos] = dst_add
        w_m[new_pos] = w_add
        tr_m[new_pos] = tr_add

        self.key, self.src, self.dst, self.w, self.trace = key_m, src_m, dst_m, w_m, tr_m

    def add_path(self, ids: torch.Tensor, mass: torch.Tensor) -> None:
        """Add edges ids[t] -> ids[t+1] for a token path."""
        ids = ids.to(device=self.device, dtype=torch.long).flatten()
        if ids.numel() < 2:
            return
        self.add_edges(ids[:-1], ids[1:], mass)

    def get_edges(self, src: torch.Tensor, dst: torch.Tensor):
        """Vectorized lookup of edges.

        Returns (w, trace, exists) aligned with the input pairs.
        """

        src = src.to(device=self.device, dtype=torch.long).flatten()
        dst = dst.to(device=self.device, dtype=torch.long).flatten()
        if src.shape != dst.shape:
            raise ValueError('src and dst must have the same shape')

        if src.numel() == 0 or self.num_edges == 0:
            w0 = torch.zeros(src.numel(), device=self.device, dtype=self.dtype)
            t0 = torch.zeros(src.numel(), device=self.device, dtype=self.dtype)
            e0 = torch.zeros(src.numel(), device=self.device, dtype=torch.bool)
            return w0, t0, e0

        src = src.clamp(min=0, max=self.num_nodes - 1)
        dst = dst.clamp(min=0, max=self.num_nodes - 1)
        key_q = src * self.num_nodes + dst

        pos = torch.searchsorted(self.key, key_q)
        in_range = pos < self.key.numel()
        pos_safe = pos.clamp(max=max(int(self.key.numel()) - 1, 0))
        hit = self.key[pos_safe] == key_q
        exists = in_range & hit

        w = torch.zeros_like(key_q, dtype=self.dtype)
        t = torch.zeros_like(key_q, dtype=self.dtype)
        if bool(exists.any()):
            p = pos_safe[exists]
            w[exists] = self.w[p]
            t[exists] = self.trace[p]
        return w, t, exists


    # ----------------------------
    # Query
    # ----------------------------

    def batch_edges(self, src_ids: torch.Tensor) -> Optional[BondBatch]:
        """Collect all outgoing edges for provided sources."""
        if src_ids.numel() == 0 or self.num_edges == 0:
            return None
        src_ids = torch.unique(src_ids.to(device=self.device, dtype=torch.long).flatten())
        src_sorted, _ = torch.sort(src_ids)
        mask = _isin_sorted(self.src, src_sorted)
        if not bool(mask.any()):
            return None
        eidx = torch.nonzero(mask, as_tuple=False).flatten()
        return BondBatch(eidx=eidx, src=self.src[eidx], dst=self.dst[eidx], w=self.w[eidx], trace=self.trace[eidx])

    # ----------------------------
    # Updates
    # ----------------------------

    def update_edges(self, eidx: torch.Tensor, w_new: torch.Tensor, trace_new: torch.Tensor) -> None:
        self.w[eidx] = w_new
        self.trace[eidx] = trace_new

    def prune_by_src_mean(self, src_ids: torch.Tensor) -> None:
        """Prune weak edges using per-source adaptive threshold (no fixed constants).

        For each source, keep edges with w >= mean(w_out(src)) - std(w_out(src)).
        Pruning is implemented by zeroing edge state; call compact() to drop zeros.
        """

        batch = self.batch_edges(src_ids)
        if batch is None:
            return

        eps = self.eps
        s = batch.src
        w = batch.w

        # Per-source mean/std over the active subset.
        s_u, inv = torch.unique(s, return_inverse=True)
        out_sum = torch.zeros(int(s_u.numel()), device=self.device, dtype=self.dtype)
        out_sum.index_add_(0, inv, w)
        count = torch.zeros(int(s_u.numel()), device=self.device, dtype=self.dtype)
        count.index_add_(0, inv, torch.ones_like(w))
        mean = out_sum / (count + eps)

        out_sq = torch.zeros(int(s_u.numel()), device=self.device, dtype=self.dtype)
        out_sq.index_add_(0, inv, w * w)
        ex2 = out_sq / (count + eps)
        var = (ex2 - mean * mean).clamp(min=0.0)
        std = torch.sqrt(var + eps)
        # Numerical slack so the minimum edge in a 2-edge fanout isn't dropped by roundoff.
        thresh = (mean - std) - eps
        keep = w >= thresh[inv]
        if bool(keep.all()):
            return

        pruned = batch.eidx[~keep]
        if pruned.numel() == 0:
            return
        self.w[pruned] = torch.zeros_like(self.w[pruned])
        self.trace[pruned] = torch.zeros_like(self.trace[pruned])

    def compact(self) -> None:
        """Physically remove zeroed edges."""
        if self.num_edges == 0:
            return
        keep = (self.w > 0) | (self.trace != 0)
        if bool(keep.all()):
            return
        self.key = self.key[keep]
        self.src = self.src[keep]
        self.dst = self.dst[keep]
        self.w = self.w[keep]
        self.trace = self.trace[keep]

    # ----------------------------
    # Flow
    # ----------------------------

    def flow_from_distribution(self, dist: torch.Tensor) -> torch.Tensor:
        """One-step flow of a distribution through the graph.

        dist: [V] nonnegative mass over sources
        returns: [V] mass over destinations
        """

        dist = dist.to(device=self.device, dtype=self.dtype).flatten()
        if self.num_edges == 0:
            return torch.zeros_like(dist)

        # Restrict to sources with nonzero mass.
        src_ids = torch.nonzero(dist > 0, as_tuple=False).flatten()
        batch = self.batch_edges(src_ids)
        if batch is None:
            return torch.zeros_like(dist)

        eps = self.eps

        s = batch.src
        d = batch.dst
        w = batch.w

        # Normalize outgoing weights per source (within the active subset).
        s_u, inv = torch.unique(s, return_inverse=True)
        out_sum = torch.zeros(int(s_u.numel()), device=self.device, dtype=self.dtype)
        out_sum.index_add_(0, inv, w)
        w_norm = w / (out_sum[inv] + eps)

        contrib = dist[s] * w_norm
        out = torch.zeros_like(dist)
        out.index_add_(0, d, contrib)
        return out
