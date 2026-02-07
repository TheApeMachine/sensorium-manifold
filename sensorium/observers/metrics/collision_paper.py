"""Collision experiment artifacts for paper-ready figures.

These are compact (small) derived arrays intended for figure projectors.
They avoid shipping full particle arrays into projectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class CollisionPaperArtifactsConfig:
    hist_max_multiplicity: int = 16
    topk_tokens: int = 24
    max_wave_points: int = 2048


class CollisionPaperArtifacts:
    """Extract compact collision + wave summaries from the final state."""

    def __init__(self, config: CollisionPaperArtifactsConfig | None = None):
        self.config = config or CollisionPaperArtifactsConfig()

    def observe(self, state: dict | None = None, **_kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}

        out: Dict[str, Any] = {}

        # ------------------------------------------------------------
        # Token collision multiplicity distribution
        # ------------------------------------------------------------
        tok = state.get("token_ids")
        hist_max = int(max(1, self.config.hist_max_multiplicity))
        xs: List[int] = list(range(1, hist_max + 1))
        ys: List[int] = [0 for _ in xs]
        top_tokens: List[int] = []
        top_counts: List[int] = []

        if tok is not None and hasattr(tok, "detach"):
            tok_i64 = tok.detach().to("cpu").to(torch.int64)
            if tok_i64.numel() > 0:
                uniq, counts = torch.unique(tok_i64, return_counts=True)
                counts_np = counts.to(torch.int64).numpy()

                ys = [int(np.sum(counts_np == k)) for k in xs]
                overflow = int(np.sum(counts_np > hist_max))
                if overflow > 0:
                    xs.append(hist_max + 1)
                    ys.append(overflow)

                # Top-k collided tokens.
                topk = int(max(1, self.config.topk_tokens))
                order = np.argsort(counts_np)[::-1][:topk]
                top_counts = counts_np[order].astype(np.int64).tolist()
                top_tokens = (
                    uniq.to(torch.int64).numpy()[order].astype(np.int64).tolist()
                )

        out.update(
            {
                "collision_mult_x": xs,
                "collision_mult_y": ys,
                "collision_top_tokens": top_tokens,
                "collision_top_counts": top_counts,
            }
        )

        # ------------------------------------------------------------
        # Wave spectrum snapshot (omega_lattice, psi_amplitude)
        # ------------------------------------------------------------
        omega = state.get("omega_lattice")
        psi_amp = state.get("psi_amplitude")
        if (
            omega is not None
            and psi_amp is not None
            and hasattr(omega, "detach")
            and hasattr(psi_amp, "detach")
        ):
            w = omega.detach().to("cpu").to(torch.float32).flatten()
            a = psi_amp.detach().to("cpu").to(torch.float32).flatten()
            if w.numel() == a.numel() and w.numel() > 0:
                n = int(w.numel())
                max_pts = int(max(32, self.config.max_wave_points))
                if n > max_pts:
                    idx = torch.linspace(0, n - 1, steps=max_pts).to(torch.int64)
                    w = w.index_select(0, idx)
                    a = a.index_select(0, idx)
                out.update(
                    {
                        "wave_omega": w.numpy().astype(np.float32).tolist(),
                        "wave_psi_amp": a.numpy().astype(np.float32).tolist(),
                    }
                )

        return out
