"""Scaling experiment datasets.

Provides various datasets for testing scaling characteristics:
- Population dynamics (repetitive text)
- Interference (multi-pattern)
- Compute scaling (random bytes)
- Generalization (natural vs synthetic text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Tuple, List

import numpy as np

from sensorium.dataset.base import BaseDataset


class ScalingTestType(Enum):
    """Types of scaling tests."""
    POPULATION = "population"
    INTERFERENCE = "interference"
    COMPUTE = "compute"
    LATENCY = "latency"
    GENERALIZATION = "generalization"


class GeneralizationType(Enum):
    """Types of data for generalization tests."""
    REPETITIVE = "repetitive"
    SEMI_RANDOM = "semi_random"
    NATURAL_LIKE = "natural_like"
    PURE_RANDOM = "pure_random"


@dataclass
class ScalingDatasetConfig:
    """Configuration for scaling datasets."""
    test_type: ScalingTestType = ScalingTestType.POPULATION
    n_bytes: int = 2000
    n_patterns: int = 1  # For interference test
    generalization_type: GeneralizationType = GeneralizationType.REPETITIVE
    seed: int = 42


# Natural text corpus for natural-like generalization tests
NATURAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The only thing we have to fear is fear itself.",
    "In the beginning was the word.",
    "It was the best of times, it was the worst of times.",
    "Ask not what your country can do for you.",
    "I think, therefore I am.",
    "The truth shall set you free.",
]


class ScalingDataset(BaseDataset):
    """Dataset for scaling experiments.
    
    Generates different types of data for various scaling tests.
    """
    
    def __init__(self, config: ScalingDatasetConfig | None = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = ScalingDatasetConfig(**kwargs)
        
        self.data: bytes = b""
        self._generate_data()
    
    def _generate_data(self):
        """Generate data based on test type and configuration."""
        if self.config.test_type == ScalingTestType.POPULATION:
            self._generate_population_data()
        elif self.config.test_type == ScalingTestType.INTERFERENCE:
            self._generate_interference_data()
        elif self.config.test_type == ScalingTestType.COMPUTE:
            self._generate_compute_data()
        elif self.config.test_type == ScalingTestType.LATENCY:
            self._generate_latency_data()
        elif self.config.test_type == ScalingTestType.GENERALIZATION:
            self._generate_generalization_data()
    
    def _generate_population_data(self):
        """Generate repetitive text for population dynamics test."""
        text = ("The quick brown fox jumps over the lazy dog. " * 50)[:self.config.n_bytes]
        self.data = text.encode("utf-8")
    
    def _generate_interference_data(self):
        """Generate multi-pattern data for interference test."""
        patterns = [f"Pattern{i:02d}XYZ" for i in range(self.config.n_patterns)]
        reps = max(1, 200 // self.config.n_patterns)
        text = " ".join(patterns * reps)
        self.data = text.encode("utf-8")[:self.config.n_bytes]
    
    def _generate_compute_data(self):
        """Generate random bytes for compute scaling test."""
        np.random.seed(self.config.seed)
        self.data = bytes(np.random.randint(0, 256, self.config.n_bytes, dtype=np.uint8))
    
    def _generate_latency_data(self):
        """Generate random bytes for latency test."""
        np.random.seed(self.config.seed)
        self.data = bytes(np.random.randint(0, 256, self.config.n_bytes, dtype=np.uint8))
    
    def _generate_generalization_data(self):
        """Generate data for generalization test based on type."""
        if self.config.generalization_type == GeneralizationType.REPETITIVE:
            text = "The cat sat on the mat. " * 100
        elif self.config.generalization_type == GeneralizationType.SEMI_RANDOM:
            text = "".join(chr(65 + (i * 7) % 26) for i in range(self.config.n_bytes))
        elif self.config.generalization_type == GeneralizationType.NATURAL_LIKE:
            text = self._get_natural_text()
        elif self.config.generalization_type == GeneralizationType.PURE_RANDOM:
            np.random.seed(self.config.seed)
            text = "".join(chr(np.random.randint(32, 127)) for _ in range(self.config.n_bytes))
        else:
            text = ""
        
        self.data = text.encode("utf-8")[:self.config.n_bytes]
    
    def _get_natural_text(self) -> str:
        """Get natural-ish text with varied sentence structures."""
        text = ""
        for i in range(50):
            text += NATURAL_SENTENCES[i % len(NATURAL_SENTENCES)] + " "
            if i % 3 == 0:
                text += NATURAL_SENTENCES[(i * 7) % len(NATURAL_SENTENCES)] + " "
        return text
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate data as (byte_value, sequence_index) tuples."""
        for idx, byte_val in enumerate(self.data):
            yield (byte_val, idx)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        return (f"ScalingDataset(type={self.config.test_type.value}, "
                f"n_bytes={len(self.data)})")
