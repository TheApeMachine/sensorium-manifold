"""Text dataset for next-token prediction experiments.

Generates text patterns with shared prefixes to test the thermodynamic trie's
ability to learn multiple continuations.

Example:
    from sensorium.dataset import TextDataset, TextDatasetConfig
    
    dataset = TextDataset(TextDatasetConfig(
        patterns=["The cat sat.", "The cat ran."],
        segment_size=16,
    ))
    
    manifold.add_dataset(dataset.generate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np

from sensorium.dataset.base import BaseDataset
from sensorium.dataset.synthetic import SyntheticDataset, SyntheticConfig, SyntheticPattern


# Default text patterns for next-token prediction
DEFAULT_TEXT_PATTERNS = [
    "The cat sat.    ",  # 16 chars
    "The cat ran.    ",  # 16 chars  
    "The cat ate.    ",  # 16 chars
    "The dog sat.    ",  # 16 chars
    "The dog ran.    ",  # 16 chars
    "A cat sat.      ",  # 16 chars (different start, same structure)
    "A dog ran.      ",  # 16 chars
]

DEFAULT_PATTERN_COUNTS = {
    "The cat sat.    ": 30,  # Most common
    "The cat ran.    ": 20,
    "The cat ate.    ": 10,
    "The dog sat.    ": 15,
    "The dog ran.    ": 10,
    "A cat sat.      ": 8,
    "A dog ran.      ": 7,
}


@dataclass
class TextDatasetConfig:
    """Configuration for text dataset.
    
    Attributes:
        patterns: List of text patterns (must all be same length)
        pattern_counts: Dict mapping patterns to their frequency counts
        segment_size: Size of each segment (must match pattern length)
        seed: Random seed for reproducibility
    """
    patterns: List[str] = field(default_factory=lambda: DEFAULT_TEXT_PATTERNS.copy())
    pattern_counts: Dict[str, int] = field(default_factory=lambda: DEFAULT_PATTERN_COUNTS.copy())
    segment_size: int = 16
    seed: int = 42


class TextDataset(BaseDataset):
    """Text dataset with multiple patterns sharing prefixes.
    
    Uses SyntheticDataset with TEXT_PREFIX pattern under the hood.
    
    Tests the thermodynamic trie's ability to:
    1. Learn multiple different continuations for the same prefix
    2. Generalize to unseen combinations
    3. Handle branching (same prefix, different suffixes)
    
    Example:
        dataset = TextDataset(TextDatasetConfig(
            patterns=["Hello world!", "Hello there!"],
            segment_size=12,
        ))
        
        for byte_val, idx in dataset.generate():
            # Process training data
            pass
    """
    
    def __init__(self, config: TextDatasetConfig | None = None, **kwargs):
        """
        Args:
            config: Dataset configuration
            **kwargs: Shortcut for config fields (patterns, pattern_counts, seed)
        """
        if config:
            self.config = config
        else:
            self.config = TextDatasetConfig(**kwargs)
        
        self._rng = np.random.RandomState(self.config.seed)
        
        # Verify all patterns are correct length
        for p in self.config.patterns:
            if len(p) != self.config.segment_size:
                raise ValueError(
                    f"Pattern '{p}' has length {len(p)}, "
                    f"expected {self.config.segment_size}"
                )
        
        # Build training dataset using SyntheticDataset
        self._train_dataset = SyntheticDataset(SyntheticConfig(
            pattern=SyntheticPattern.TEXT_PREFIX,
            text_patterns=self.config.patterns,
            pattern_counts=self.config.pattern_counts,
            seed=self.config.seed,
        ))
        
        # Build test dataset (uniform sampling)
        self._test_dataset = SyntheticDataset(SyntheticConfig(
            pattern=SyntheticPattern.TEXT_PREFIX,
            text_patterns=self.config.patterns,
            num_units=len(self.config.patterns) * 3,  # 3 of each
            seed=self.config.seed + 1,  # Different seed for test
        ))
        
        # Pre-compute bytes for convenience
        self.train_bytes = b"".join(self._train_dataset.get_patterns())
        self.test_bytes = b"".join(self._test_dataset.get_patterns())
        self.train_text = self.train_bytes.decode("utf-8")
        self.test_text = self.test_bytes.decode("utf-8")
    
    @property
    def segment_size(self) -> int:
        """Return the segment size."""
        return self.config.segment_size
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples."""
        return self._train_dataset.generate()
    
    def generate_test(self) -> Iterator[Tuple[int, int]]:
        """Generate test data as (byte_value, sequence_index) tuples."""
        return self._test_dataset.generate()
    
    def train_test_split(self) -> Tuple[bytes, bytes]:
        """Return pre-built train/test split as bytes."""
        return self.train_bytes, self.test_bytes
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Return pattern frequency statistics."""
        return self.config.pattern_counts.copy()
    
    def __repr__(self) -> str:
        return (
            f"TextDataset(patterns={len(self.config.patterns)}, "
            f"segment_size={self.config.segment_size})"
        )
