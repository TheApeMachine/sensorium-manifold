"""Spectral dataset for FFT-based data processing.

This dataset performs spectral analysis (FFT/STFT) on audio or other
time-series data before tokenization. It's useful for:

- Cocktail party separation (multiple speakers)
- Audio classification/generation
- Any scenario where frequency decomposition aids learning

The key insight is that the manifold can cluster coherent spectral
patterns (frequencies that "belong together") into carriers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, Optional
import numpy as np

from sensorium.dataset.base import BaseDataset, SpectralConfig


class SpectralDataset(BaseDataset):
    """Dataset that transforms audio/signals through FFT before tokenization.
    
    Performs STFT on input audio and yields frequency bin information as
    (byte_value, sequence_index) tuples, where:
    
    - byte_value: Quantized magnitude (0-255)
    - sequence_index: Frequency bin index (0 to num_bins-1)
    
    This allows the manifold to learn spectral structure, clustering
    frequencies that co-occur (e.g., from the same speaker).
    
    Examples:
        # Load audio for spectral analysis
        dataset = SpectralDataset(SpectralConfig(
            path=Path("./audio/mixed.wav"),
            fft_size=1024,
            hop_size=256,
            sample_rate=22050,
        ))
        
        # Process with phase information
        dataset = SpectralDataset(SpectralConfig(
            path=Path("./audio/mixed.wav"),
            include_phase=True,
        ))
    """
    
    # WAV format constants
    WAV_HEADER_SIZE = 44
    
    def __init__(self, config: SpectralConfig):
        self.config = config
        
        # Computed data (lazily populated)
        self._stft_frames: Optional[np.ndarray] = None
        self._magnitudes: Optional[np.ndarray] = None
        self._phases: Optional[np.ndarray] = None
        self._samples: Optional[np.ndarray] = None
    
    def _load_wav(self, path: Path) -> np.ndarray:
        """Load WAV file and return audio samples as float32.
        
        Only supports 16-bit mono WAV files.
        """
        with open(path, "rb") as f:
            # Skip header
            f.seek(self.WAV_HEADER_SIZE)
            raw = f.read()
        
        # Convert 16-bit samples to float32
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]
        return samples
    
    def _compute_stft(self, samples: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform.
        
        Returns complex array of shape (num_frames, num_bins).
        """
        fft_size = self.config.fft_size
        hop_size = self.config.hop_size
        
        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fft_size) / fft_size))
        
        # Compute number of frames
        num_samples = len(samples)
        num_frames = (num_samples - fft_size) // hop_size + 1
        
        # Allocate output
        stft = np.zeros((num_frames, fft_size // 2 + 1), dtype=np.complex64)
        
        # Compute STFT
        for i in range(num_frames):
            start = i * hop_size
            frame = samples[start:start + fft_size] * window
            stft[i] = np.fft.rfft(frame)
        
        return stft
    
    def _process_audio(self):
        """Load and process audio file."""
        if self._stft_frames is not None:
            return  # Already processed
        
        path = self.config.path
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Load audio samples
        self._samples = self._load_wav(path)
        
        # Compute STFT
        self._stft_frames = self._compute_stft(self._samples)
        self._magnitudes = np.abs(self._stft_frames)
        self._phases = np.angle(self._stft_frames)
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples from spectral data.
        
        For each STFT frame:
        - byte_value: Quantized magnitude (0-255) of the frequency bin
        - sequence_index: Frequency bin index (0 to num_bins-1)
        
        If include_phase is True, phase values are interleaved:
        - Odd sequence indices: magnitude values
        - Even sequence indices: phase values (also quantized to 0-255)
        """
        self._process_audio()
        
        n_frames, n_bins = self._magnitudes.shape
        threshold = self._magnitudes.max() * self.config.magnitude_threshold
        
        # Quantize magnitudes to byte range
        mag_max = self._magnitudes.max() + 1e-10
        
        for frame_idx in range(n_frames):
            for bin_idx in range(n_bins):
                magnitude = self._magnitudes[frame_idx, bin_idx]
                
                # Skip below threshold
                if magnitude < threshold:
                    continue
                
                # Quantize magnitude to [0, 255]
                mag_byte = int((magnitude / mag_max) * 255)
                mag_byte = min(255, max(0, mag_byte))
                
                if self.config.include_phase:
                    # Yield magnitude then phase
                    yield (mag_byte, bin_idx * 2)
                    
                    # Quantize phase from [-pi, pi] to [0, 255]
                    phase = self._phases[frame_idx, bin_idx]
                    phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
                    phase_byte = int(phase_normalized * 255)
                    phase_byte = min(255, max(0, phase_byte))
                    yield (phase_byte, bin_idx * 2 + 1)
                else:
                    yield (mag_byte, bin_idx)
    
    # =========================================================================
    # Convenience methods for reconstruction
    # =========================================================================
    
    def get_stft_frames(self) -> np.ndarray:
        """Return the computed STFT frames."""
        self._process_audio()
        return self._stft_frames.copy()
    
    def get_samples(self) -> np.ndarray:
        """Return the original audio samples."""
        self._process_audio()
        return self._samples.copy()
    
    def get_stft_shape(self) -> Tuple[int, int]:
        """Return (num_frames, num_bins) shape."""
        self._process_audio()
        return self._stft_frames.shape
    
    def reconstruct_from_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct audio from masked STFT.
        
        Args:
            mask: Boolean or float array of shape (num_frames, num_bins)
                  True/1.0 means keep this bin, False/0.0 means zero it out
        
        Returns:
            Reconstructed audio samples as float32 array
        """
        self._process_audio()
        
        # Apply mask to STFT
        masked_stft = self._stft_frames * mask
        
        # Inverse STFT (overlap-add)
        fft_size = self.config.fft_size
        hop_size = self.config.hop_size
        n_frames = masked_stft.shape[0]
        
        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fft_size) / fft_size))
        
        # Output length
        out_len = (n_frames - 1) * hop_size + fft_size
        output = np.zeros(out_len, dtype=np.float32)
        window_sum = np.zeros(out_len, dtype=np.float32)
        
        for i in range(n_frames):
            frame = np.fft.irfft(masked_stft[i], n=fft_size)
            start = i * hop_size
            output[start:start + fft_size] += frame * window
            window_sum[start:start + fft_size] += window ** 2
        
        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-10)
        output = output / window_sum
        
        # Match original length
        if self._samples is not None:
            output = output[:len(self._samples)]
        
        return output
    
    def save_wav(self, samples: np.ndarray, path: Path):
        """Save audio samples to WAV file.
        
        Args:
            samples: Float32 audio samples in [-1, 1] range
            path: Output file path
        """
        import struct
        
        # Convert to 16-bit integers
        samples_16 = (samples * 32767).astype(np.int16)
        raw_data = samples_16.tobytes()
        
        # Write WAV file
        with open(path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(raw_data)))
            f.write(b"WAVE")
            
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # Chunk size
            f.write(struct.pack("<H", 1))   # Audio format (PCM)
            f.write(struct.pack("<H", 1))   # Num channels
            f.write(struct.pack("<I", self.config.sample_rate))  # Sample rate
            f.write(struct.pack("<I", self.config.sample_rate * 2))  # Byte rate
            f.write(struct.pack("<H", 2))   # Block align
            f.write(struct.pack("<H", 16))  # Bits per sample
            
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", len(raw_data)))
            f.write(raw_data)
