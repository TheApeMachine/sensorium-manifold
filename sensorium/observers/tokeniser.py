import numpy as np

class TokeniserObserver:
    def __init__(self, *, segment_size: int, hash_prime: int, vocab_size: int | None = None):
        self.segment_size = int(segment_size)
        self.hash_prime = int(hash_prime)
        self.vocab_size = int(vocab_size) if vocab_size is not None else None

    def observe(self, **kwargs):
        """Compute simple collision statistics for the (byte,index) hash."""
        vocab_size = self.vocab_size if self.vocab_size is not None else int(state["vocab_size"])
        if (vocab_size & (vocab_size - 1)) != 0:
            raise ValueError("vocab_size must be power-of-two (fast-path assumption).")
        mask = int(vocab_size - 1)

        n = int(state["data"].shape[0])

        if n == 0:
            return {"n": 0.0, "unique_ids": 0.0, "collision_rate": 0.0}

        segment_size = int(state.get("segment_size", self.segment_size))
        pos = (np.arange(n, dtype=np.int64) % segment_size).astype(np.int64)
        tid = ((state["data"].astype(np.int64) * int(self.hash_prime) + pos) & mask).astype(np.int64)
        unique = int(np.unique(tid).shape[0])
        collisions = n - unique

        return {
            "n": float(n),
            "unique_ids": float(unique),
            "collision_rate": float(collisions) / float(n),
        }