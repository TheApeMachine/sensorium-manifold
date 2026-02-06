"""Map-vs-path metrics for the post-hash (semantic) story.

These metrics are intentionally "measurement only": no physics changes, no kernels.
They turn the final simulation state into:
- a compressed MAP view (folded mass over keys)
- a lossless PATH view (transitions over keys within each sample)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def _byte_label(b: int) -> str:
    b = int(b) & 0xFF
    if 32 <= b <= 126 and b not in (92,):  # printable; avoid "\" in tables
        return chr(b)
    return f"0x{b:02x}"


def _as_cpu_np_int64(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().to("cpu").to(torch.int64).numpy()
    return None


def _as_cpu_np_float64(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().to("cpu").to(torch.float64).numpy()
    return None


def _fold_mass_over_keys(keys: np.ndarray, masses: np.ndarray) -> tuple[float, float, float]:
    """Return (top1_share, participation_ratio, entropy_bits)."""
    if keys.size == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(keys, kind="stable")
    k_sorted = keys[order]
    m_sorted = masses[order]
    _, start = np.unique(k_sorted, return_index=True)
    ms = np.add.reduceat(m_sorted, start)
    tot = float(ms.sum())
    if not (tot > 0.0):
        return 0.0, 0.0, 0.0
    top1 = float(np.max(ms) / tot)
    pr = float((tot * tot) / float(np.sum(ms * ms)))
    p = ms / tot
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    return top1, pr, H


def _transition_metrics_from_keys(
    *,
    keys: np.ndarray,
    sample_indices: np.ndarray,
    sequence_indices: np.ndarray,
    topk: int = 20,
) -> tuple[int, int, float, float, list[dict]]:
    """Return (edges, unique_edges, top1_prob, entropy_bits, top_edges_decoded)."""
    if keys.size == 0:
        return 0, 0, 0.0, 0.0, []

    # Vectorized: sort once by (sample, seq), then take adjacency with Δt=+1.
    order = np.lexsort((sequence_indices, sample_indices))
    s = sample_indices[order]
    t = sequence_indices[order]
    k = keys[order]
    same = (s[1:] == s[:-1])
    next_pos = (t[1:] == (t[:-1] + 1))
    ok = same & next_pos
    src = k[:-1][ok]
    dst = k[1:][ok]
    if src.size == 0:
        return 0, 0, 0.0, 0.0, []

    e_all = (src.astype(np.uint64) << np.uint64(32)) | (dst.astype(np.uint64) & np.uint64(0xFFFFFFFF))
    e_uniq, e_cnt = np.unique(e_all, return_counts=True)
    total = float(e_cnt.sum())
    top1_prob = float(float(np.max(e_cnt)) / total) if total > 0 else 0.0
    pe = e_cnt.astype(np.float64) / max(total, 1.0)
    H = float(-(pe * np.log2(pe + 1e-300)).sum())

    # Decode top-k for artifacts.
    ord2 = np.argsort(e_cnt)[::-1][: int(max(0, topk))]
    top_edges = []
    for j in ord2:
        ek = int(e_uniq[j])
        src32 = (ek >> 32) & 0xFFFFFFFF
        dst32 = ek & 0xFFFFFFFF
        src_pos = (src32 >> 8) & 0xFFFFFFFF
        dst_pos = (dst32 >> 8) & 0xFFFFFFFF
        src_b = src32 & 0xFF
        dst_b = dst32 & 0xFF
        top_edges.append(
            {
                "src": {"pos": int(src_pos), "byte": int(src_b), "label": _byte_label(src_b)},
                "dst": {"pos": int(dst_pos), "byte": int(dst_b), "label": _byte_label(dst_b)},
                "count": int(e_cnt[j]),
            }
        )

    return int(e_all.size), int(e_uniq.size), float(top1_prob), float(H), top_edges


@dataclass(frozen=True, slots=True)
class KeySpec:
    """How to construct a per-particle key."""

    kind: str  # "sequence_byte" or "spatial_morton_byte"


class MapPathMetrics:
    """Compute folding + transitions over a key (map vs path)."""

    def __init__(self, *, key: KeySpec, topk: int = 20):
        self.key = key
        self.topk = int(topk)

    def observe(self, state: dict | None = None, **kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}

        b = _as_cpu_np_int64(state.get("byte_values"))
        t = _as_cpu_np_int64(state.get("sequence_indices"))
        s = _as_cpu_np_int64(state.get("sample_indices"))
        m = _as_cpu_np_float64(state.get("masses"))
        if b is None or t is None or s is None:
            return {}
        n = int(b.size)
        if n == 0:
            return {"n_tokens": 0}
        if not (t.size == n and s.size == n):
            return {}

        # Construct keys.
        if self.key.kind == "sequence_byte":
            key = ((t.astype(np.uint64) & np.uint64(0xFFFFFFFF)) << np.uint64(8)) | (b.astype(np.uint64) & np.uint64(0xFF))
        elif self.key.kind == "spatial_morton_byte":
            # Prefer precomputed key from the physics domain if available.
            pre = _as_cpu_np_int64(state.get("spatial_token_ids"))
            if pre is not None and pre.size == n:
                key = pre.astype(np.uint64)
            else:
                # Fallback: compute Morton cell id from positions and grid dims.
                pos = state.get("positions", None)
                dx = state.get("dx", None)
                grid_dims = kwargs.get("grid_dims", None)
                if pos is None or not hasattr(pos, "detach"):
                    return {}
                if not isinstance(dx, (float, int)) or float(dx) <= 0.0:
                    return {}
                if not (isinstance(grid_dims, (tuple, list)) and len(grid_dims) == 3):
                    return {}
                gx, gy, gz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
                p = pos.detach().to("cpu").to(torch.float32).numpy()
                if p.shape[0] != n:
                    return {}
                inv_dx = 1.0 / float(dx)
                ix = (np.floor(p[:, 0] * inv_dx).astype(np.int64) % gx).astype(np.uint32)
                iy = (np.floor(p[:, 1] * inv_dx).astype(np.int64) % gy).astype(np.uint32)
                iz = (np.floor(p[:, 2] * inv_dx).astype(np.int64) % gz).astype(np.uint32)

                def _part1by2(v: np.ndarray) -> np.ndarray:
                    v = v & np.uint32(0x3FF)  # up to 10 bits per axis (covers grid<=1024)
                    v = (v | (v << np.uint32(16))) & np.uint32(0x030000FF)
                    v = (v | (v << np.uint32(8))) & np.uint32(0x0300F00F)
                    v = (v | (v << np.uint32(4))) & np.uint32(0x030C30C3)
                    v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
                    return v

                morton = (_part1by2(ix) | (_part1by2(iy) << np.uint32(1)) | (_part1by2(iz) << np.uint32(2))).astype(np.uint64)
                key = (morton << np.uint64(8)) | (b.astype(np.uint64) & np.uint64(0xFF))
        else:
            return {}

        unique_keys = int(np.unique(key).size)
        key_collision_rate = float(1.0 - (float(unique_keys) / float(n)))

        out: Dict[str, Any] = {
            "n_tokens": int(n),
            "n_samples": int(np.max(s) + 1) if s.size else 0,
            "unique_keys": int(unique_keys),
            "key_collision_rate": float(key_collision_rate),
        }

        # Folding requires masses.
        if m is not None and m.size == n:
            top1, pr, H = _fold_mass_over_keys(key, m)
            out.update({"fold_top1": float(top1), "fold_pr": float(pr), "fold_entropy": float(H)})

        # Path transitions.
        edges, uniq_e, top1_p, edge_H, top_edges = _transition_metrics_from_keys(
            keys=key, sample_indices=s, sequence_indices=t, topk=self.topk
        )
        out.update(
            {
                "transition_edges": int(edges),
                "transition_unique_edges": int(uniq_e),
                "transition_top1_prob": float(top1_p),
                "transition_entropy": float(edge_H),
                # For projector to write a per-run artifact.
                "transitions_top_edges": top_edges,
            }
        )

        return out

