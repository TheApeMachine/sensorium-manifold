"""Cross-modal dataset for unified text-image processing.

Converts images to frequency-domain bytes and combines with text labels
for cross-modal manifold experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, List, Dict

import numpy as np
import torch

from sensorium.dataset.base import BaseDataset


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal dataset."""
    top_k_freq: int = 64
    image_size: int = 32


class CrossModalDataset(BaseDataset):
    """Dataset that converts images to frequency-domain bytes with text labels.
    
    Uses 2D FFT to convert images to frequency space, then quantizes
    the top-k magnitude/phase pairs to bytes alongside text tokens.
    """
    
    def __init__(
        self,
        image: np.ndarray,
        text_labels: List[str],
        config: CrossModalConfig | None = None,
        **kwargs
    ):
        if config:
            self.config = config
        else:
            self.config = CrossModalConfig(**kwargs)
        
        self.image = image
        self.text_labels = text_labels
        
        # Convert image to frequency bytes
        self.image_bytes, self.metadata = self._image_to_frequency_bytes(image)
        self.text_bytes = self._text_to_bytes(text_labels)
        self.combined_data = self.image_bytes + self.text_bytes
    
    def _image_to_frequency_bytes(self, image: np.ndarray) -> Tuple[bytes, Dict]:
        """Convert image to frequency-domain bytes."""
        image_t = torch.tensor(image, dtype=torch.float32)
        spectrum = torch.fft.fft2(image_t)
        spectrum_shifted = torch.fft.fftshift(spectrum)
        
        magnitude = torch.abs(spectrum_shifted)
        phase = torch.angle(spectrum_shifted)
        
        mag_flat = magnitude.flatten()
        phase_flat = phase.flatten()
        
        if self.config.top_k_freq < len(mag_flat):
            topk_indices = torch.topk(mag_flat, self.config.top_k_freq).indices
        else:
            topk_indices = torch.arange(len(mag_flat))
        
        mag_selected = mag_flat[topk_indices]
        phase_selected = phase_flat[topk_indices]
        
        mag_max = mag_selected.max()
        if mag_max > 0:
            mag_norm = (mag_selected / mag_max * 255).to(torch.uint8)
        else:
            mag_norm = torch.zeros_like(mag_selected, dtype=torch.uint8)
        
        phase_norm = ((phase_selected + np.pi) / (2 * np.pi) * 255).to(torch.uint8)
        combined = torch.stack([mag_norm, phase_norm], dim=1).flatten()
        
        metadata = {
            "shape": image.shape,
            "topk_indices": topk_indices.numpy(),
            "mag_max": mag_max.item(),
            "spectrum_shape": spectrum_shifted.shape,
            "mag_selected": mag_selected.numpy(),
            "phase_selected": phase_selected.numpy(),
        }
        
        return bytes(combined.numpy()), metadata
    
    def _text_to_bytes(self, labels: List[str]) -> bytes:
        """Convert text labels to bytes."""
        text = " ".join(labels)
        return text.encode("utf-8")
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate combined data as (byte_value, sequence_index) tuples."""
        for idx, byte_val in enumerate(self.combined_data):
            yield (byte_val, idx)
    
    def get_frequency_space(self) -> Dict[str, np.ndarray]:
        """Extract frequency-space coordinates for visualization.
        
        Returns:
            Dict with 'u', 'v' frequency coordinates and 'energy' values.
        """
        H, W = self.metadata.get("shape", (32, 32))
        topk_indices = self.metadata.get("topk_indices", np.array([]))
        mag_selected = self.metadata.get("mag_selected", None)
        
        if mag_selected is None or len(topk_indices) == 0:
            return {"u": np.array([]), "v": np.array([]), "energy": np.array([])}
        
        n_freq = min(len(topk_indices), len(mag_selected))
        u_coords, v_coords, energies = [], [], []
        
        for i, idx in enumerate(topk_indices[:n_freq]):
            u = (idx // W) - H // 2
            v = (idx % W) - W // 2
            energy = float(mag_selected[i])
            u_coords.append(u)
            v_coords.append(v)
            energies.append(energy)
        
        return {"u": np.array(u_coords), "v": np.array(v_coords), "energy": np.array(energies)}
    
    def __repr__(self) -> str:
        return (f"CrossModalDataset(image_shape={self.image.shape}, "
                f"labels={self.text_labels}, top_k={self.config.top_k_freq})")


def create_stripe_image(size: int, orientation: str) -> np.ndarray:
    """Create a stripe pattern image."""
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    
    if orientation == "horizontal":
        image = 0.5 + 0.4 * np.sin(yy * 2)
    elif orientation == "vertical":
        image = 0.5 + 0.4 * np.sin(xx * 2)
    elif orientation == "diagonal":
        image = 0.5 + 0.4 * np.sin((xx + yy) * 1.5)
    else:
        image = 0.5 * np.ones((size, size))
    
    return image.astype(np.float32)


def create_checkerboard_image(size: int) -> np.ndarray:
    """Create a checkerboard pattern."""
    image = np.zeros((size, size), dtype=np.float32)
    block_size = size // 4
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                image[i:i+block_size, j:j+block_size] = 1.0
    return image
