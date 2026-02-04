"""Kernel MNIST bytes: vectorized manifold ingest.

This experiment loads MNIST image data using the fast numpy path.
Each byte becomes one "token" in the carrier dynamics. Position resets
every 784 bytes (one image), so bytes at the same pixel position across
different images hash consistently.
"""
from __future__ import annotations

import numpy as np

from sensorium.experiments.base import Experiment
from sensorium.manifold import Manifold, ManifoldConfig
from sensorium.dataset.filesystem import FilesystemDataset
from sensorium.dataset.base import DatasetConfig


# MNIST binary format constants
IMAGE_HEADER_SIZE = 16  # magic(4) + count(4) + rows(4) + cols(4)
IMAGE_SIZE = 28 * 28    # 784 bytes per image


class KernelMNISTBytes(Experiment):

    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        self.data_dir = self.repo_root / "data" / "mnist"
        
        # Dataset handles loading - segment_size is in DatasetConfig
        self.dataset = FilesystemDataset(
            config=DatasetConfig(
                path=self.data_dir,
                name="mnist",
                num_classes=10,
                num_examples_per_class=60000,
                segment_size=IMAGE_SIZE,
                header_size=IMAGE_HEADER_SIZE,
            )
        )
        self.cfg = ManifoldConfig(
            generator=self.dataset.generate,
            dashboard=False,
        )

    def observe(self, state: dict):
        print(state)

    def run(self):
        """Run vectorized ingest and return the final manifold state."""
        self.observe(Manifold(self.cfg).run())
