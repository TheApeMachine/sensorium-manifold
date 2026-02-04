from __future__ import annotations

from typing import Iterator
import random

from sensorium.dataset.base import BaseDataset, DatasetConfig


class SyntheticDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.vocab = config.vocab if config.vocab is not None else [bytes([i]) for i in range(256)]

    def generate(self) -> Iterator[bytes]:
        for i in range(self.config.num_classes):
            for j in range(self.config.num_examples_per_class):
                yield random.choice(self.vocab) * self.config.segment_size
                yield b"\x00" * self.config.segment_size