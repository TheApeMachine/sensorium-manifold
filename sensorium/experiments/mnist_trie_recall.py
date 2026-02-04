"""MNIST trie recall experiment (paper-ready).

This keeps the experiment file small by delegating:
- MNIST parsing + splits to `sensorium.dataset.mnist_idx`
- Recall metrics + plotting to `sensorium.observers.mnist_trie`
"""

from __future__ import annotations

import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    SpectralSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.carrier import CarrierObserver
from sensorium.dataset.filesystem import FilesystemDataset
from sensorium.dataset.base import DatasetConfig
from sensorium.dataset.mnist_idx import (
    MNIST_IMAGE_SIZE,
)
from sensorium.observers.energy import EnergyObserver


class MNISTTrieRecallExperiment(Experiment):
    """MNIST holdout recall from a compressed thermodynamic trie."""

    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        self.data_dir = self.repo_root / "data" / "mnist"
        self.train = FilesystemDataset(
            config=DatasetConfig(
                path=self.data_dir / "MNIST" / "raw" / "train-images-idx3-ubyte",
                header_size=16,
                limit=100,  # Reduced for byte-level ingestion
                segment_size=784,
            )
        )
        self.holdout = FilesystemDataset(
            config=DatasetConfig(
                path=self.data_dir / "MNIST" / "raw" / "t10k-images-idx3-ubyte",
                header_size=16,
                offset=1000,
                limit=5,
                segment_size=784,
            )
        )

        self.manifold = Manifold(
            SimulationConfig(
                dashboard=True,
                generator=None,
                geometric=GeometricSimulationConfig(grid_size=(32, 32, 32), dt=0.01),
                spectral=SpectralSimulationConfig(grid_size=(32, 32, 32), dt=0.01),
                tokenizer=TokenizerConfig(
                    hash_vocab_size=4096,
                    hash_prime=31,
                ),
            ),
            observers={
                "spectral": InferenceObserver([CarrierObserver(None)]),
            },
        )

    def observe(self, manifold):
        # We generate visualizations at the end.
        state = manifold.state
        token_ids = state.get("token_ids")
        energies = state.get("energies")
        
        if token_ids is None or energies is None:
            print("Warning: Missing required state fields")
            return
        
        # Dehash token_ids back to bytes
        # Hash formula: token_id = (byte * prime + pos) & mask
        # Inverse: byte = ((token_id - pos) * inv_prime) & mask
        prime = 31
        vocab = 4096
        mask = vocab - 1
        segment_size = 784
        inv_prime = pow(prime, -1, vocab)
        
        if isinstance(token_ids, torch.Tensor):
            token_ids_t = token_ids
            device = token_ids_t.device
        else:
            token_ids_t = torch.tensor(token_ids, dtype=torch.int64)
            device = torch.device("cpu")
        
        # Calculate positions (wrapping every segment_size bytes)
        n = len(token_ids_t)
        indices = torch.arange(n, device=device, dtype=torch.int64)
        pos = torch.remainder(indices, segment_size)
        
        # Dehash: byte = ((token_id - pos) * inv_prime) & mask
        # Note: We need to handle the subtraction carefully to avoid negative values
        # The formula: token_id = (byte * prime + pos) & mask
        # Reverse: (token_id - pos) mod vocab = (byte * prime) mod vocab
        # So: byte = ((token_id - pos) mod vocab * inv_prime) mod vocab
        
        # Handle subtraction with proper wrapping
        diff = token_ids_t - pos
        target = diff & mask  # This handles negative values correctly due to two's complement
        
        recovered_vals = (target * inv_prime) & mask
        
        # Filter valid bytes (should be < 256)
        valid_mask = recovered_vals < 256
        num_invalid = (~valid_mask).sum().item()
        if num_invalid > 0:
            print(f"Warning: {num_invalid} invalid bytes recovered (>= 256) out of {n} total")
        
        # Use all recovered values, but clamp invalid ones to 0-255 range
        recovered_vals_clamped = torch.clamp(recovered_vals, 0, 255)
        prompt_flat = recovered_vals_clamped.cpu().numpy().astype(np.uint8)
        
        # Debug: check first few values
        if n > 0:
            print(f"First 10 token_ids: {token_ids_t[:10].cpu().numpy()}")
            print(f"First 10 positions: {pos[:10].cpu().numpy()}")
            print(f"First 10 recovered bytes: {prompt_flat[:10]}")
            print(f"Recovered bytes range: [{prompt_flat.min()}, {prompt_flat.max()}]")
        
        # Convert energies to numpy
        energy_by_tid = energies.cpu().numpy() if isinstance(energies, torch.Tensor) else np.array(energies)
        
        # Create observer and set required attributes
        observer = EnergyObserver(
            prime=31,
            vocab=4096,
            MNIST_IMAGE_SIZE=784,
        )
        observer.prompt_flat = prompt_flat
        observer.prompt_len = 784  # Full image for now
        observer.energy_by_tid = energy_by_tid
        
        out = observer.observe()

        print(f"Reconstructed image shape: {out.shape}, dtype: {out.dtype}, min: {out.min()}, max: {out.max()}")

        # Save as PNG using PIL
        try:
            from PIL import Image
            # Ensure values are in [0, 255] range
            img_array = np.clip(out, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale
            img.save("out.png")
            print(f"Saved image to out.png")
        except ImportError:
            # Fallback to matplotlib if PIL not available
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.imsave("out.png", out, cmap='gray', vmin=0, vmax=255)
                plt.close()
                print(f"Saved image to out.png (using matplotlib)")
            except ImportError:
                print("Warning: Neither PIL nor matplotlib available, cannot save image")
                print(f"Image array: shape={out.shape}, dtype={out.dtype}")

    def run(self):
        idx = 0

        for dataset in [self.train, self.holdout]:
            self.manifold.set_generator(dataset.generate)
            self.manifold.run(settle=idx == 1, inference=idx == 1)
            idx += 1

        self.observe(self.manifold)
