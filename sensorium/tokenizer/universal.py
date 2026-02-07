"""Universal Tokenizer.

Adapter between `DatasetProtocol.generate()` (yielding `(byte_value, sequence_index)`)
and the Manifold Loader contract.

Loader expects `tokenizer.stream()` to yield batches shaped like:

  {
    "oscillator": {token_ids, sequence_indices, phase, omega, energy},
    "particle":   {masses, heats, energies, excitations, byte_values}
  }

All values are torch.Tensors of length N.

Token id scheme:
  token_id = (sequence_index << 8) | byte_value
so identical bytes at identical relative positions collide by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from sensorium.kernels.runtime import get_device
from sensorium.tokenizer.prototype import Tokenizer
from sensorium.dataset.base import DatasetProtocol
from sensorium.console import console


@dataclass(frozen=True, slots=True)
class UniversalTokenizerConfig:
    max_tokens: int = 65536
    batch_tokens: int = 65536
    energy_init: float = 1.0
    omega_min: float = -4.0
    omega_max: float = 4.0
    seed: int = 0


class UniversalTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        datasets: List[DatasetProtocol],
        config: Optional[UniversalTokenizerConfig] = None,
    ) -> None:
        self.datasets = datasets
        self.config = config or UniversalTokenizerConfig()
        self.device = torch.device(get_device())

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config.seed))

    def stream(self) -> Iterator[dict]:
        console.info("Streaming tokens")

        cfg = self.config
        max_tokens = int(cfg.max_tokens)
        batch_tokens = int(cfg.batch_tokens)
        if max_tokens <= 0 or batch_tokens <= 0:
            return

        buf_bytes: list[int] = []
        buf_seq: list[int] = []
        emitted = 0

        for dataset in self.datasets:
            for byte_val, seq_idx in dataset.generate():
                if emitted >= max_tokens:
                    break

                buf_bytes.append(int(byte_val) & 0xFF)
                buf_seq.append(int(seq_idx))
                emitted += 1

                if len(buf_bytes) >= batch_tokens:
                    yield self._make_batch(buf_bytes, buf_seq)
                    buf_bytes = []
                    buf_seq = []

            if emitted >= max_tokens:
                break

        if buf_bytes:
            yield self._make_batch(buf_bytes, buf_seq)

    def _make_batch(self, byte_values: list[int], sequence_indices: list[int]) -> dict:
        if len(byte_values) != len(sequence_indices):
            raise ValueError("byte_values and sequence_indices must have same length")
        n = len(byte_values)
        if n == 0:
            return {"oscillator": {}, "particle": {}}

        dev = self.device
        f32 = torch.float32

        byte_i64 = torch.tensor(byte_values, device=dev, dtype=torch.int64)
        seq_i64 = torch.tensor(sequence_indices, device=dev, dtype=torch.int64)

        token_ids = (seq_i64 << 8) | byte_i64

        phase = torch.rand((n,), generator=self._rng, device="cpu", dtype=f32).to(dev)
        phase = phase * float(2.0 * torch.pi)

        b01 = byte_i64.to(torch.float32) / 255.0
        omega = (
            float(self.config.omega_min)
            + (float(self.config.omega_max) - float(self.config.omega_min)) * b01
        )
        energy = torch.full((n,), float(self.config.energy_init), device=dev, dtype=f32)

        masses = torch.ones((n,), device=dev, dtype=f32)
        heats = torch.zeros((n,), device=dev, dtype=f32)
        energies = energy.clone()
        excitations = omega.clone()

        return {
            "oscillator": {
                "token_ids": token_ids,
                "sequence_indices": seq_i64,
                "phase": phase,
                "omega": omega.to(f32),
                "energy": energy,
            },
            "particle": {
                "byte_values": byte_i64.to(torch.float32),
                "masses": masses,
                "heats": heats,
                "energies": energies,
                "excitations": excitations,
            },
        }
