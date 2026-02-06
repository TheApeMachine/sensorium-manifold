"""Cocktail party audio dataset for speaker separation.

Loads WAV files and converts to time-frequency representation
using Short-Time Fourier Transform (STFT).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Optional

import numpy as np

from sensorium.dataset.base import BaseDataset


@dataclass
class CocktailPartyConfig:
    """Configuration for cocktail party dataset."""
    wav_path: Path = Path("sensorium/experiments/two_speakers.wav")
    fft_size: int = 1024
    hop_size: int = 256
    sample_rate: int = 22050
    magnitude_threshold_ratio: float = 0.01


class CocktailPartyDataset(BaseDataset):
    """Audio dataset with STFT for speaker separation experiments.
    
    Loads a WAV file, computes STFT, and yields time-frequency bins
    as (byte_value, sequence_index) tuples.
    """
    
    WAV_HEADER_SIZE = 44
    
    def __init__(self, config: CocktailPartyConfig | None = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = CocktailPartyConfig(**kwargs)
        
        self.audio_samples: Optional[np.ndarray] = None
        self.stft_frames: Optional[np.ndarray] = None
        self.magnitudes: Optional[np.ndarray] = None
        self.phases: Optional[np.ndarray] = None
        self.active_mask: Optional[np.ndarray] = None
        self.frame_indices: Optional[np.ndarray] = None
        self.bin_indices: Optional[np.ndarray] = None
        self.active_magnitudes: Optional[np.ndarray] = None
        self.active_phases: Optional[np.ndarray] = None
        
        self._load()
    
    def _load(self):
        """Load and process the WAV file."""
        if not self.config.wav_path.exists():
            print(f"Warning: WAV file not found at {self.config.wav_path}")
            return
        
        # Load audio
        self.audio_samples = self._load_wav(self.config.wav_path)
        
        # Compute STFT
        self.stft_frames = self._compute_stft(self.audio_samples)
        self.magnitudes = np.abs(self.stft_frames)
        self.phases = np.angle(self.stft_frames)
        
        # Create active mask
        threshold = self.magnitudes.max() * self.config.magnitude_threshold_ratio
        self.active_mask = self.magnitudes > threshold
        
        # Get active indices
        self.frame_indices, self.bin_indices = np.where(self.active_mask)
        self.active_magnitudes = self.magnitudes[self.active_mask]
        self.active_phases = self.phases[self.active_mask]
        
        n_samples = len(self.audio_samples)
        duration = n_samples / self.config.sample_rate
        print(f"Loaded {n_samples:,} samples ({duration:.2f}s)")
        print(f"STFT: {self.stft_frames.shape[0]} frames x {self.stft_frames.shape[1]} bins")
        print(f"Active bins: {len(self.frame_indices):,}")
    
    def _load_wav(self, path: Path) -> np.ndarray:
        """Load WAV file and return samples as float32 array."""
        with open(path, "rb") as f:
            data = f.read()
        
        audio_bytes = data[self.WAV_HEADER_SIZE:]
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
    
    def _compute_stft(self, samples: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform."""
        n_samples = len(samples)
        n_frames = (n_samples - self.config.fft_size) // self.config.hop_size + 1
        window = np.hanning(self.config.fft_size)
        n_bins = self.config.fft_size // 2 + 1
        stft = np.zeros((n_frames, n_bins), dtype=np.complex64)
        
        for i in range(n_frames):
            start = i * self.config.hop_size
            frame = samples[start:start + self.config.fft_size] * window
            stft[i] = np.fft.rfft(frame)
        
        return stft
    
    def istft(self, stft: np.ndarray) -> np.ndarray:
        """Inverse Short-Time Fourier Transform."""
        n_frames, n_bins = stft.shape
        n_samples = (n_frames - 1) * self.config.hop_size + self.config.fft_size
        output = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)
        window = np.hanning(self.config.fft_size)
        
        for i in range(n_frames):
            start = i * self.config.hop_size
            frame = np.fft.irfft(stft[i], n=self.config.fft_size)
            output[start:start + self.config.fft_size] += frame * window
            window_sum[start:start + self.config.fft_size] += window ** 2
        
        window_sum = np.maximum(window_sum, 1e-8)
        output /= window_sum
        return output
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate frequency data as (byte_value, sequence_index) tuples.
        
        Each active time-frequency bin is converted to a byte based on
        its normalized magnitude.
        """
        if self.active_magnitudes is None:
            return
        
        # Normalize magnitudes to bytes
        max_mag = self.active_magnitudes.max()
        if max_mag > 0:
            byte_vals = (self.active_magnitudes / max_mag * 255).astype(np.uint8)
        else:
            byte_vals = np.zeros(len(self.active_magnitudes), dtype=np.uint8)
        
        for idx, byte_val in enumerate(byte_vals):
            yield (int(byte_val), idx)
    
    @property
    def n_frames(self) -> int:
        return self.stft_frames.shape[0] if self.stft_frames is not None else 0
    
    @property
    def n_bins(self) -> int:
        return self.stft_frames.shape[1] if self.stft_frames is not None else 0
    
    @property
    def n_particles(self) -> int:
        return len(self.frame_indices) if self.frame_indices is not None else 0
    
    def reconstruct_separated_audio(
        self, 
        labels: np.ndarray, 
        num_speakers: int = 2
    ) -> list[np.ndarray]:
        """Reconstruct separated audio streams using STFT masking.
        
        Args:
            labels: Cluster label for each active bin.
            num_speakers: Number of speakers to separate.
        
        Returns:
            List of audio arrays, one per speaker.
        """
        separated = []
        
        for speaker_id in range(num_speakers):
            # Create a mask for this speaker
            speaker_mask = np.zeros((self.n_frames, self.n_bins), dtype=np.float32)
            
            speaker_bin_mask = labels == speaker_id
            speaker_frames = self.frame_indices[speaker_bin_mask]
            speaker_bins = self.bin_indices[speaker_bin_mask]
            speaker_mask[speaker_frames, speaker_bins] = 1.0
            
            # Apply mask to STFT
            masked_stft = self.stft_frames * speaker_mask
            
            # Inverse STFT
            speaker_audio = self.istft(masked_stft)
            
            # Normalize
            max_val = np.abs(speaker_audio).max()
            if max_val > 0:
                speaker_audio = speaker_audio / max_val * 0.9
            
            separated.append(speaker_audio)
        
        return separated
    
    def __repr__(self) -> str:
        return f"CocktailPartyDataset(path={self.config.wav_path}, n_particles={self.n_particles})"


def save_wav(samples: np.ndarray, path: Path, sample_rate: int = 22050):
    """Save audio samples as WAV file."""
    import struct
    
    samples_int16 = (samples * 32767).astype(np.int16)
    audio_bytes = samples_int16.tobytes()
    data_size = len(audio_bytes)
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        1,  # num channels
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b'data',
        data_size
    )
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(audio_bytes)
