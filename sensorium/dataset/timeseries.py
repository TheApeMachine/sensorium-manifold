"""Time series dataset.

Generates various time series patterns and quantizes them to bytes for
the universal tokenizer.

Example:
    from sensorium.dataset import TimeSeriesDataset, TimeSeriesConfig
    
    dataset = TimeSeriesDataset(TimeSeriesConfig(
        length=2000,
        series_type="periodic",
        train_ratio=0.8,
    ))
    
    manifold.add_dataset(dataset.generate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

import numpy as np

from sensorium.dataset.base import BaseDataset


@dataclass
class TimeSeriesConfig:
    """Configuration for time series dataset.
    
    Attributes:
        length: Total length of the time series
        series_type: Type of signal to generate
        train_ratio: Fraction of data for training (rest is test)
        segment_size: Segment size for index reset (captures periodicity)
        seed: Random seed for reproducibility
    """
    length: int = 2000
    series_type: Literal["periodic", "trend_seasonal", "regime_switch", "sawtooth"] = "periodic"
    train_ratio: float = 0.8
    segment_size: int = 50  # Match dominant period
    seed: int = 42


def quantize_to_bytes(values: np.ndarray) -> Tuple[bytes, float, float]:
    """Quantize float values to bytes (0-255).
    
    Args:
        values: Float array to quantize
    
    Returns:
        Tuple of (quantized_bytes, min_val, max_val)
    """
    min_val = float(values.min())
    max_val = float(values.max())
    normalized = (values - min_val) / (max_val - min_val + 1e-10)
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return bytes(quantized), min_val, max_val


class TimeSeriesDataset(BaseDataset):
    """Dataset for time series with various signal types.
    
    Generates synthetic time series data and quantizes to bytes.
    Supports multiple signal types for testing different patterns.
    
    Example:
        dataset = TimeSeriesDataset(TimeSeriesConfig(
            series_type="periodic",
            length=2000,
        ))
        
        for byte_val, idx in dataset.generate():
            # Process training data
            pass
    """
    
    def __init__(self, config: TimeSeriesConfig | None = None, **kwargs):
        """
        Args:
            config: Dataset configuration
            **kwargs: Shortcut for config fields
        """
        if config:
            self.config = config
        else:
            self.config = TimeSeriesConfig(**kwargs)
        
        self._rng = np.random.RandomState(self.config.seed)
        
        # Generate signal
        self.values = self._generate()
        self.bytes_data, self.min_val, self.max_val = quantize_to_bytes(self.values)
        
        # Split train/test
        split_idx = int(self.config.length * self.config.train_ratio)
        self.train_bytes = self.bytes_data[:split_idx]
        self.test_bytes = self.bytes_data[split_idx:]
        self.split_idx = split_idx
    
    def _generate(self) -> np.ndarray:
        """Generate the time series based on type."""
        t = np.arange(self.config.length, dtype=np.float32)
        series_type = self.config.series_type
        
        if series_type == "periodic":
            # Multi-frequency periodic signal
            y = (
                50 * np.sin(2 * np.pi * t / 50) +
                20 * np.sin(2 * np.pi * t / 13) +
                10 * np.sin(2 * np.pi * t / 7) +
                5 * self._rng.randn(self.config.length)
            )
        elif series_type == "trend_seasonal":
            # Linear trend with seasonal component
            trend = 0.05 * t
            seasonal = 30 * np.sin(2 * np.pi * t / 100)
            noise = 5 * self._rng.randn(self.config.length)
            y = trend + seasonal + noise
        elif series_type == "regime_switch":
            # Frequency changes halfway through
            mid = self.config.length // 2
            y = np.zeros(self.config.length, dtype=np.float32)
            y[:mid] = 50 * np.sin(2 * np.pi * t[:mid] / 100)
            y[mid:] = 50 * np.sin(2 * np.pi * t[mid:] / 20)
            y += 5 * self._rng.randn(self.config.length)
        elif series_type == "sawtooth":
            # Sawtooth wave
            period = 50
            y = (t % period) / period * 100 + 5 * self._rng.randn(self.config.length)
        else:
            raise ValueError(f"Unknown series type: {series_type}")
        
        return y
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples.
        
        Index resets every segment_size bytes to capture periodicity.
        """
        segment_size = self.config.segment_size
        for idx, byte_val in enumerate(self.train_bytes):
            yield (byte_val, idx % segment_size)
    
    def generate_test(self) -> Iterator[Tuple[int, int]]:
        """Generate test data as (byte_value, sequence_index) tuples."""
        segment_size = self.config.segment_size
        for idx, byte_val in enumerate(self.test_bytes):
            # Continue index from where training left off
            global_idx = self.split_idx + idx
            yield (byte_val, global_idx % segment_size)
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataset(type={self.config.series_type!r}, "
            f"length={self.config.length}, segment_size={self.config.segment_size})"
        )
