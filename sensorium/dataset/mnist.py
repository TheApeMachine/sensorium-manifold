"""MNIST dataset loader for image experiments.

Loads MNIST images as raw bytes for manifold processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, List

import numpy as np

from sensorium.dataset.base import BaseDataset


MNIST_IMAGE_SIZE = 28 * 28  # 784 pixels


@dataclass
class MNISTConfig:
    """Configuration for MNIST dataset."""
    data_dir: Path = Path("data/mnist")
    train: bool = True
    limit: int | None = None


class MNISTDataset(BaseDataset):
    """Load MNIST images as raw bytes.
    
    Images are loaded from IDX format or generated synthetically
    if files are not found.
    """
    
    def __init__(self, config: MNISTConfig | None = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = MNISTConfig(**kwargs)
        
        # MNIST file paths
        if self.config.train:
            images_path = self.config.data_dir / "MNIST" / "raw" / "train-images-idx3-ubyte"
            labels_path = self.config.data_dir / "MNIST" / "raw" / "train-labels-idx1-ubyte"
        else:
            images_path = self.config.data_dir / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
            labels_path = self.config.data_dir / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"
        
        self.images_path = images_path
        self.labels_path = labels_path
        self.images: List[bytes] = []
        self.labels: List[int] = []
        
        self._load()
    
    def _load(self):
        """Load MNIST from IDX format."""
        if not self.images_path.exists():
            print(f"Warning: MNIST not found at {self.images_path}")
            self._generate_synthetic()
            return
        
        # Read images
        with open(self.images_path, 'rb') as f:
            f.read(16)  # Skip header
            data = f.read()
        
        n_images = len(data) // MNIST_IMAGE_SIZE
        if self.config.limit:
            n_images = min(n_images, self.config.limit)
        
        for i in range(n_images):
            start = i * MNIST_IMAGE_SIZE
            end = start + MNIST_IMAGE_SIZE
            self.images.append(data[start:end])
        
        # Read labels
        if self.labels_path.exists():
            with open(self.labels_path, 'rb') as f:
                f.read(8)  # Skip header
                label_data = f.read()
            self.labels = list(label_data[:n_images])
        else:
            self.labels = [0] * n_images
        
        print(f"Loaded {len(self.images)} MNIST images")
    
    def _generate_synthetic(self):
        """Generate synthetic digit-like images for testing."""
        print("Generating synthetic digit-like images...")
        rng = np.random.RandomState(42)
        n_images = self.config.limit or 100
        
        for digit in range(10):
            for _ in range(n_images // 10):
                img = np.zeros(MNIST_IMAGE_SIZE, dtype=np.uint8)
                
                if digit == 0:
                    # Circle
                    for y in range(28):
                        for x in range(28):
                            dist = np.sqrt((x - 14)**2 + (y - 14)**2)
                            if 5 < dist < 8:
                                img[y * 28 + x] = 255
                elif digit == 1:
                    # Vertical line
                    for y in range(6, 22):
                        img[y * 28 + 14] = 255
                        img[y * 28 + 15] = 255
                else:
                    # Random pattern with some structure
                    base = digit * 25
                    for y in range(4 + digit, 24 - digit):
                        for x in range(4 + digit, 24 - digit):
                            if rng.rand() < 0.3:
                                img[y * 28 + x] = base + rng.randint(0, 100)
                
                noise = rng.randint(0, 30, size=MNIST_IMAGE_SIZE).astype(np.uint8)
                img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
                
                self.images.append(bytes(img))
                self.labels.append(digit)
        
        print(f"Generated {len(self.images)} synthetic images")
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Yield (byte_value, sequence_index) tuples.
        
        The sequence index resets at each image boundary.
        """
        for img in self.images:
            for idx, byte_val in enumerate(img):
                yield (byte_val, idx)
    
    def __repr__(self) -> str:
        return f"MNISTDataset(train={self.config.train}, images={len(self.images)})"
