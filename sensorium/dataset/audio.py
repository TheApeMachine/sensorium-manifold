"""Audio waveform dataset.

Generates synthetic audio waveforms and quantizes them to bytes.

Example:
    from sensorium.dataset import AudioDataset, AudioDatasetConfig
    
    dataset = AudioDataset(AudioDatasetConfig(
        waveform_type="sine",
        frequency=440.0,
    ))
    
    manifold.add_dataset(dataset.generate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

import numpy as np

from sensorium.dataset.base import BaseDataset


def quantize_audio_to_bytes(samples: np.ndarray) -> bytes:
    """Quantize float audio samples (-1 to 1) to bytes (0-255)."""
    normalized = (samples + 1.0) / 2.0
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return bytes(quantized)


def dequantize_bytes_to_audio(data: bytes) -> np.ndarray:
    """Dequantize bytes back to float audio samples."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return (arr / 255.0) * 2.0 - 1.0


@dataclass
class AudioDatasetConfig:
    """Configuration for audio dataset.
    
    Attributes:
        waveform_type: Type of waveform to generate
        sample_rate: Audio sample rate in Hz
        duration: Duration in seconds
        frequency: Base frequency in Hz
        train_ratio: Fraction of data for training
        seed: Random seed
    """
    waveform_type: Literal["sine", "square", "sawtooth", "mixed"] = "sine"
    sample_rate: int = 8000
    duration: float = 1.0
    frequency: float = 440.0
    train_ratio: float = 0.8
    seed: int = 42


class AudioDataset(BaseDataset):
    """Dataset for audio waveform with various waveform types.
    
    Generates synthetic audio waveforms (sine, square, sawtooth, mixed)
    and quantizes them to bytes for the universal tokenizer.
    
    Example:
        dataset = AudioDataset(AudioDatasetConfig(
            waveform_type="sine",
            frequency=440.0,
            duration=1.0,
        ))
        
        for byte_val, idx in dataset.generate():
            # Process training data
            pass
    """
    
    def __init__(self, config: AudioDatasetConfig | None = None, **kwargs):
        """
        Args:
            config: Dataset configuration
            **kwargs: Shortcut for config fields
        """
        if config:
            self.config = config
        else:
            self.config = AudioDatasetConfig(**kwargs)
        
        # Period in samples
        self.period_samples = int(self.config.sample_rate / self.config.frequency)
        
        # Generate waveform
        samples = self._generate()
        self.audio_bytes = quantize_audio_to_bytes(samples)
        
        # Split train/test
        split_idx = int(len(self.audio_bytes) * self.config.train_ratio)
        self.train_bytes = self.audio_bytes[:split_idx]
        self.test_bytes = self.audio_bytes[split_idx:]
        self.split_idx = split_idx
    
    def _generate(self) -> np.ndarray:
        """Generate synthetic waveform."""
        n_samples = int(self.config.duration * self.config.sample_rate)
        t = np.arange(n_samples, dtype=np.float32) / self.config.sample_rate
        rng = np.random.RandomState(self.config.seed)
        freq = self.config.frequency
        
        if self.config.waveform_type == "sine":
            y = np.sin(2 * np.pi * freq * t)
        elif self.config.waveform_type == "square":
            y = np.sign(np.sin(2 * np.pi * freq * t))
        elif self.config.waveform_type == "sawtooth":
            period = 1.0 / freq
            y = 2 * (t / period - np.floor(t / period + 0.5))
        elif self.config.waveform_type == "mixed":
            y = (0.5 * np.sin(2 * np.pi * freq * t) +
                 0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                 0.2 * np.sin(2 * np.pi * freq * 3 * t))
        else:
            y = np.zeros(n_samples)
        
        # Add slight noise
        y += 0.02 * rng.randn(n_samples)
        return np.clip(y, -1, 1).astype(np.float32)
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples.
        
        Index resets every period_samples to capture periodicity.
        """
        for idx, byte_val in enumerate(self.train_bytes):
            yield (byte_val, idx % self.period_samples)
    
    def generate_test(self) -> Iterator[Tuple[int, int]]:
        """Generate test data as (byte_value, sequence_index) tuples."""
        for idx, byte_val in enumerate(self.test_bytes):
            global_idx = self.split_idx + idx
            yield (byte_val, global_idx % self.period_samples)
    
    def __repr__(self) -> str:
        return (
            f"AudioDataset(type={self.config.waveform_type!r}, "
            f"freq={self.config.frequency}Hz, period={self.period_samples})"
        )
