"""Kernel-based continuous simulation (finite run).

This is the kernel analogue of the interactive simulator, but runs for a fixed
number of steps so it can be used in an experiment pipeline.

Outputs (best-effort):
- `paper/figures/continuous_final.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

from sensorium.experiments.base import Experiment
from sensorium.manifold import Manifold
from sensorium.manifold import ManifoldConfig


class KernelContinuous(Experiment):
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        self.cfg = ManifoldConfig(
            generator=self.load(),
            ready="crystallized",
            dashboard=self.dashboard,
            video_path=self.video_path,
        )

    def load(self) -> Iterator[Tuple[Path, bytes]]:
        for i in range(10):
            yield None, bytes([i])

    def observe(self, state: dict):
        pass

    def run(self):
        # self.observe(Manifold(self.cfg).run())
        pass
