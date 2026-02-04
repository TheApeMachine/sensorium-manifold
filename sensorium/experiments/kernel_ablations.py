"""Kernel-based ablation study (Metal/MPS).

Writes:
- `paper/tables/ablation.tex`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

from sensorium.manifold import Manifold, ManifoldConfig
from .base import Experiment


class KernelAblations(Experiment):
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        self.data_dir = Path("../../data/mnist")
        self.cfg = ManifoldConfig(
            chunk_size=1,
            generator=self.load(), 
            ready="crystallized"
        )

    def load(self) -> Iterator[Tuple[Path, bytes]]:
        for i in range(10):
            yield None, bytes([i])

    def observe(self, state: dict):
        pass

    def run(self):
        # """Run generator-driven ingest and return the final manifold state."""
        # self.observe(Manifold(self.cfg).run())
        pass
