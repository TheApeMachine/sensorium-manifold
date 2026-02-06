"""Token distribution metrics (hash-space bookkeeping).

These are NOT "semantic compression" metrics; they describe the hash-ID distribution.
They are kept to help explain reviewer confusion and to provide baselines.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


class TokenDistributionMetrics:
    """Compute unique-id ratio, collision rate, and entropy of token_ids."""

    def observe(self, state: dict | None = None, **kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        tok = state.get("token_ids", None)
        if tok is None or not hasattr(tok, "detach"):
            return {}

        tok_np = tok.detach().to("cpu").to(torch.int64).numpy()
        n = int(tok_np.size)
        if n == 0:
            return {"n_unique_tokens": 0, "compression_ratio": 1.0, "collision_rate_observed": 0.0, "entropy": 0.0}

        n_unique = int(np.unique(tok_np).size)
        compression_ratio = float(n_unique / n)
        collision_rate_observed = float(1.0 - compression_ratio)

        vocab = kwargs.get("hash_vocab_size", None)
        if isinstance(vocab, int) and vocab > 0:
            minlength = int(vocab)
        else:
            minlength = int(max(1, tok_np.max(initial=0) + 1))
        counts = np.bincount(tok_np, minlength=minlength)
        total = float(counts.sum())
        if total > 0.0:
            p = counts.astype(np.float64) / total
            p = p[p > 0]
            entropy = float(-(p * np.log2(p)).sum())
        else:
            entropy = 0.0

        return {
            "n_unique_tokens": n_unique,
            "compression_ratio": compression_ratio,
            "collision_rate_observed": collision_rate_observed,
            "entropy": entropy,
        }

