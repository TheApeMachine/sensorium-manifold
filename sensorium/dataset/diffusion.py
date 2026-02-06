"""Text diffusion dataset.

Dataset for text denoising/inpainting experiments using repetitive patterns.

Example:
    from sensorium.dataset import DiffusionDataset, DiffusionDatasetConfig
    
    dataset = DiffusionDataset(DiffusionDatasetConfig(
        max_bytes=2000,
        train_ratio=0.8,
    ))
    
    manifold.add_dataset(dataset.generate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

from sensorium.dataset.base import BaseDataset


# Default sample text with repetitive patterns
DEFAULT_SAMPLE_TEXT = """The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
"""


@dataclass
class DiffusionDatasetConfig:
    """Configuration for diffusion dataset.
    
    Attributes:
        text: Source text (uses default if None)
        max_bytes: Maximum bytes to use from text
        train_ratio: Fraction of data for training
        segment_size: Segment size for index reset
        seed: Random seed
    """
    text: str | None = None
    max_bytes: int = 2000
    train_ratio: float = 0.8
    segment_size: int = 64
    seed: int = 42


class DiffusionDataset(BaseDataset):
    """Dataset for text diffusion/denoising experiments.
    
    Uses repetitive text patterns to enable pattern learning.
    The manifold learns byte patterns that can be used to
    reconstruct masked/noisy characters.
    
    Example:
        dataset = DiffusionDataset(DiffusionDatasetConfig(
            max_bytes=2000,
            segment_size=64,
        ))
        
        # Get train/test bytes
        train, test = dataset.train_bytes, dataset.test_bytes
    """
    
    def __init__(self, config: DiffusionDatasetConfig | None = None, **kwargs):
        """
        Args:
            config: Dataset configuration
            **kwargs: Shortcut for config fields
        """
        if config:
            self.config = config
        else:
            self.config = DiffusionDatasetConfig(**kwargs)
        
        # Get source text
        text = self.config.text or DEFAULT_SAMPLE_TEXT
        text_bytes = text.encode("utf-8")[:self.config.max_bytes]
        
        # Split into train/test
        split_idx = int(len(text_bytes) * self.config.train_ratio)
        self.train_bytes = text_bytes[:split_idx]
        self.test_bytes = text_bytes[split_idx:]
        self.split_idx = split_idx
    
    @property
    def segment_size(self) -> int:
        """Return the segment size."""
        return self.config.segment_size
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples."""
        segment_size = self.config.segment_size
        for idx, byte_val in enumerate(self.train_bytes):
            yield (byte_val, idx % segment_size)
    
    def generate_test(self) -> Iterator[Tuple[int, int]]:
        """Generate test data as (byte_value, sequence_index) tuples."""
        segment_size = self.config.segment_size
        for idx, byte_val in enumerate(self.test_bytes):
            yield (byte_val, idx % segment_size)
    
    def __repr__(self) -> str:
        return (
            f"DiffusionDataset(train={len(self.train_bytes)}, "
            f"test={len(self.test_bytes)}, segment_size={self.config.segment_size})"
        )
