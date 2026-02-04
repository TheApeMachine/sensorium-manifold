"""Universal tokenizer: (byte, position) → token_id via hash.

This module does *not* step physics directly.
Instead it yields "token batches" that can be used to initialize/append both:
- a geometric particle view (positions/velocities/mass/heat/energy/excitation)
- a spectral oscillator view (phase/frequency/energy)

Why: particles and oscillators are indexed views of the same entities.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, TypedDict

import torch


@dataclass
class TokenizerConfig:
    hash_vocab_size: int = 4096  # Must be power-of-two
    hash_prime: int = 31
    device: str = "mps"
    generator: Optional[Callable[[], Iterator[bytes]]] = None
    segment_size: Optional[int] = None  # Reset index every N bytes


class OscillatorTokenBatch(TypedDict):
    """Spectral-layer initializer for newly ingested tokens."""

    token_ids: torch.Tensor     # (N,) int64
    omega: torch.Tensor         # (N,) float32
    phase: torch.Tensor         # (N,) float32
    energy: torch.Tensor        # (N,) float32  (interpreted as E_osc)


class ParticleTokenBatch(TypedDict):
    """Geometric-layer initializer for newly ingested tokens.

    NOTE: This intentionally does not choose positions. Position/velocity
    injection is a policy decision of the dataset/observer/driver.
    """

    excitations: torch.Tensor   # (N,) float32  (intrinsic ω, used by spectral layer too)
    energies: torch.Tensor      # (N,) float32  (E_osc; shared with oscillator energy)
    heats: torch.Tensor         # (N,) float32  (Q)
    masses: torch.Tensor        # (N,) float32


class TokenBatch(TypedDict):
    """A single ingestion batch containing both indexed views."""

    oscillator: OscillatorTokenBatch
    particle: ParticleTokenBatch


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        vocab = int(config.hash_vocab_size)

        if vocab <= 0 or (vocab & (vocab - 1)) != 0:
            raise ValueError(
                "TokenizerConfig.hash_vocab_size must be a power-of-two (e.g., 4096, 2048, 1024, ...)"
            )

        self._mask = vocab - 1
        self._prime = config.hash_prime
        self._index = 0
        self.device = config.device
    
    def reset_index(self):
        """Reset sequence index to 0 (call at segment boundaries)."""
        self._index = 0
    
    def tokenize(self, data: bytes) -> torch.Tensor:
        """Tokenize bytes into token IDs tensor."""
        n = len(data)
        byte_t = torch.tensor(list(data), device=self.device, dtype=torch.int64)
        pos = torch.arange(
            self._index, self._index + n, device=self.device, dtype=torch.int64
        )
        seg = self.config.segment_size

        if seg:
            seg_i = int(seg)
            if seg_i <= 0:
                raise ValueError("TokenizerConfig.segment_size must be > 0 if set")

            pos = torch.remainder(pos, seg_i)
            self._index = (self._index + n) % seg_i
        else:
            self._index += n
        return (byte_t * self._prime + pos) & self._mask
    
    def stream(self) -> Iterator[TokenBatch]:
        """Yield batches of newly ingested tokens (both particle + oscillator views)."""
        if self.config.generator is None:
            raise ValueError("No generator configured")
        
        for chunk in self.config.generator():
            token_ids = self.tokenize(chunk)
            
            # [CHOICE] ω mapping from token_id
            # [FORMULA] ω = token_id * (2 / V)
            # [REASON] deterministic, monotone map into a bounded ω range; keeps ω scale O(1)
            # [NOTES] Downstream physics should treat ω as "intrinsic frequency label".
            vocab = float(self.config.hash_vocab_size)
            omega = token_ids.to(dtype=torch.float32) * (2.0 / vocab)

            # Default spectral initial state (driver may override).
            phase = torch.zeros_like(omega)
            energy = torch.ones_like(omega)

            # Default geometric properties (driver may override).
            heats = torch.zeros_like(omega)
            masses = torch.ones_like(omega)

            yield {
                "oscillator": {
                    "token_ids": token_ids,
                    "omega": omega,
                    "phase": phase,
                    "energy": energy,
                },
                "particle": {
                    # By construction this matches the oscillator ω and energy.
                    "excitations": omega,
                    "energies": energy,
                    "heats": heats,
                    "masses": masses,
                },
            }
