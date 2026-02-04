"""Kernel image handling via Universal Tokenizer (MNIST inpainting demo).

We demonstrate "native image handling" in the sense of the Universal Tokenizer:
pixels are bytes, hashed with position, with completion by sampling bytes.

NON-CHEATING DESIGN:
====================
This experiment uses a proper train/test split of MNIST images:
- Training: Learn from training images (full, unmasked)
- Inference: Given partially masked test images, reconstruct missing pixels
- No access to ground truth during reconstruction

The key mechanism is that similar images share token IDs at the same positions,
creating carrier patterns that can fill in missing pixels.

Writes:
- `paper/tables/image_gen_summary.tex`
- `paper/figures/image_gen.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any

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


MNIST_IMAGE_SIZE = 28 * 28  # 784 pixels


class MNISTLoader:
    """Load MNIST images as raw bytes."""
    
    def __init__(self, data_dir: Path, train: bool = True, limit: int = None):
        self.data_dir = data_dir
        self.train = train
        self.limit = limit
        
        # MNIST file paths
        if train:
            images_path = data_dir / "MNIST" / "raw" / "train-images-idx3-ubyte"
            labels_path = data_dir / "MNIST" / "raw" / "train-labels-idx1-ubyte"
        else:
            images_path = data_dir / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
            labels_path = data_dir / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"
        
        self.images_path = images_path
        self.labels_path = labels_path
        self.images: List[bytes] = []
        self.labels: List[int] = []
        
        self._load()
    
    def _load(self):
        """Load MNIST from IDX format."""
        if not self.images_path.exists():
            print(f"Warning: MNIST not found at {self.images_path}")
            # Generate synthetic "digit-like" images
            self._generate_synthetic()
            return
        
        # Read images
        with open(self.images_path, 'rb') as f:
            # Skip header (16 bytes: magic, num_images, rows, cols)
            f.read(16)
            data = f.read()
        
        n_images = len(data) // MNIST_IMAGE_SIZE
        if self.limit:
            n_images = min(n_images, self.limit)
        
        for i in range(n_images):
            start = i * MNIST_IMAGE_SIZE
            end = start + MNIST_IMAGE_SIZE
            self.images.append(data[start:end])
        
        # Read labels (optional, for stratified sampling)
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
        n_images = self.limit or 100
        
        for digit in range(10):
            for _ in range(n_images // 10):
                # Create a simple pattern based on digit
                img = np.zeros(MNIST_IMAGE_SIZE, dtype=np.uint8)
                
                # Draw simple shapes representing digits
                # This is very crude but gives repeatable patterns
                center = 14 * 28 + 14  # Center of 28x28
                
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
                
                # Add noise
                noise = rng.randint(0, 30, size=MNIST_IMAGE_SIZE).astype(np.uint8)
                img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
                
                self.images.append(bytes(img))
                self.labels.append(digit)
        
        print(f"Generated {len(self.images)} synthetic images")
    
    def generate(self) -> Iterator[bytes]:
        """Yield images as bytes."""
        for img in self.images:
            yield img


class ImageInpainter:
    """Reconstruct masked pixels using dual-domain inference.
    
    For images, we leverage:
    - Geometric: spatial locality (nearby pixels cluster together)
    - Spectral: carriers couple similar pixels at same positions across images
    """
    
    def __init__(self, vocab_size: int = 4096, prime: int = 31, image_size: int = 784):
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask = vocab_size - 1
        self.image_size = image_size
        self.image_width = int(np.sqrt(image_size))  # Assuming square
        self.inference = None
    
    def learn_from_manifold(self, geo_state: Dict, spec_state: Dict):
        """Set up dual-domain inference from manifold state."""
        from sensorium.observers.dual_domain import DualDomainInference
        
        self.inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=spec_state,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
    
    def inpaint(
        self,
        corrupted: bytes,
        mask_positions: List[int],
    ) -> bytes:
        """Reconstruct masked pixels using dual-domain inference.
        
        Strategy:
        1. Find neighboring unmasked pixels (geometric locality)
        2. Find carriers they couple to (spectral)
        3. Score candidates by coupling + neighbor similarity
        """
        result = bytearray(corrupted)
        
        if self.inference is None:
            return bytes(result)
        
        # Process each masked position
        for pos in mask_positions:
            # Get unmasked neighbors for context
            neighbors = self._get_neighbors(pos)
            context_positions = [n for n in neighbors if n not in mask_positions]
            
            if context_positions:
                context_indices = torch.tensor(
                    context_positions,
                    device=self.inference.device,
                    dtype=torch.int64
                )
                
                # Get carrier-based scores
                carrier_scores = self.inference.score_candidate_bytes(
                    context_indices=context_indices,
                    target_position=pos % self.image_size,  # Position within image
                    segment_size=self.image_size,
                )
            else:
                carrier_scores = np.ones(256, dtype=np.float32) / 256
            
            # Add spatial smoothness prior (similar to neighbors)
            neighbor_scores = np.zeros(256, dtype=np.float32)
            for npos in neighbors:
                if npos not in mask_positions and 0 <= npos < len(result):
                    neighbor_val = result[npos]
                    # Gaussian kernel around neighbor value
                    for pval in range(256):
                        diff = abs(pval - neighbor_val)
                        neighbor_scores[pval] += np.exp(-diff**2 / (2 * 30**2))  # σ=30
            
            if neighbor_scores.sum() > 0:
                neighbor_scores /= neighbor_scores.sum()
            
            # Combined score: carrier coupling + spatial smoothness
            combined_scores = 0.6 * carrier_scores + 0.4 * neighbor_scores
            
            result[pos] = int(np.argmax(combined_scores))
        
        return bytes(result)
    
    def _get_neighbors(self, pos: int) -> List[int]:
        """Get 4-connected neighbors in image."""
        row = pos // self.image_width
        col = pos % self.image_width
        neighbors = []
        
        if row > 0:
            neighbors.append((row - 1) * self.image_width + col)
        if row < self.image_width - 1:
            neighbors.append((row + 1) * self.image_width + col)
        if col > 0:
            neighbors.append(row * self.image_width + (col - 1))
        if col < self.image_width - 1:
            neighbors.append(row * self.image_width + (col + 1))
        
        return neighbors


class KernelImageGen(Experiment):
    """MNIST inpainting experiment using Universal Tokenizer."""
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        
        self.train_images = 100
        self.test_images = 20
        self.hash_vocab_size = 4096
        self.mask_fracs = [0.1, 0.2, 0.3, 0.5]  # Test multiple mask levels
        
        self.results: Dict[float, Dict[str, Any]] = {}
        self.examples: List[Dict[str, Any]] = []

    def observe(self, state: dict):
        """Generate paper artifacts."""
        if not self.results:
            print("Warning: No results collected")
            return
        
        import matplotlib.pyplot as plt
        
        # Summary table
        summary = {}
        for mask_frac, res in self.results.items():
            key = f"mask_{int(mask_frac*100)}pct"
            summary[f"{key}_psnr"] = res["psnr"]
            summary[f"{key}_mae"] = res["mae"]
            summary[f"{key}_mse"] = res["mse"]
        
        self.write_kv_table("image_gen_summary", summary)
        
        # Figure: Grid of examples
        n_examples = min(len(self.examples), 8)
        if n_examples == 0:
            print("Warning: No examples to visualize")
            return
        
        fig, axes = plt.subplots(n_examples, 4, figsize=(12, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, example in enumerate(self.examples[:n_examples]):
            # Original
            ax = axes[i, 0]
            original = np.array(list(example["original"])).reshape(28, 28)
            ax.imshow(original, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"Original (digit {example.get('label', '?')})")
            ax.axis('off')
            
            # Masked
            ax = axes[i, 1]
            masked = np.array(list(example["masked"])).reshape(28, 28)
            ax.imshow(masked, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"Masked ({example['mask_frac']*100:.0f}%)")
            ax.axis('off')
            
            # Reconstructed
            ax = axes[i, 2]
            recon = np.array(list(example["reconstructed"])).reshape(28, 28)
            ax.imshow(recon, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"Reconstructed")
            ax.axis('off')
            
            # Error map
            ax = axes[i, 3]
            error = np.abs(original.astype(np.float32) - recon.astype(np.float32))
            ax.imshow(error, cmap='hot', vmin=0, vmax=128)
            ax.set_title(f"Error (MAE={example['mae']:.1f})")
            ax.axis('off')
        
        plt.suptitle('MNIST Inpainting via Universal Tokenizer\n'
                    '(Left: Original, Middle-Left: Masked, Middle-Right: Reconstructed, Right: Error)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "image_gen.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/image_gen_summary.tex")

    def run(self):
        """Run MNIST inpainting experiment."""
        print("[image_gen] Starting experiment...")
        
        # Load MNIST
        data_dir = self.repo_root / "data" / "mnist"
        train_loader = MNISTLoader(data_dir, train=True, limit=self.train_images)
        test_loader = MNISTLoader(data_dir, train=False, limit=self.test_images)
        
        train_images = train_loader.images
        test_images = test_loader.images
        test_labels = test_loader.labels
        
        print(f"[image_gen] Train: {len(train_images)}, Test: {len(test_images)}")
        
        # Train manifold on training images
        tokenizer_config = TokenizerConfig(
            hash_vocab_size=4096,
            hash_prime=31,
            segment_size=MNIST_IMAGE_SIZE,  # Reset position per image
        )
        
        def train_generator():
            for img in train_images:
                for b in img:
                    yield bytes([b])
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=False,
                generator=train_generator,
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                ),
                spectral=SpectralSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                ),
                tokenizer=tokenizer_config,
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "spectral": InferenceObserver([CarrierObserver(None)])
            }
        )
        
        state = manifold.run()
        
        # Set up dual-domain inference
        geo_state = {
            "positions": state.get("positions"),
            "velocities": state.get("velocities"),
            "energies": state.get("energies"),
            "heats": state.get("heats"),
            "excitations": state.get("excitations"),
            "token_ids": state.get("token_ids"),
            "masses": state.get("masses"),
        }
        carriers = manifold.carriers or {}
        
        inpainter = ImageInpainter(
            vocab_size=4096, prime=31, image_size=MNIST_IMAGE_SIZE
        )
        inpainter.learn_from_manifold(geo_state, carriers)
        
        # Test inpainting at different mask levels
        rng = np.random.RandomState(42)
        
        for mask_frac in self.mask_fracs:
            print(f"[image_gen] Testing mask fraction: {mask_frac}")
            
            all_maes = []
            all_mses = []
            all_psnrs = []
            
            for img_idx, img in enumerate(test_images):
                # Create mask
                n_mask = int(MNIST_IMAGE_SIZE * mask_frac)
                mask_positions = list(rng.choice(MNIST_IMAGE_SIZE, size=n_mask, replace=False))
                
                # Apply mask
                masked = bytearray(img)
                for pos in mask_positions:
                    masked[pos] = 128  # Gray as mask value
                
                # Inpaint
                reconstructed = inpainter.inpaint(bytes(masked), mask_positions)
                
                # Calculate metrics
                original_np = np.array(list(img), dtype=np.float32)
                recon_np = np.array(list(reconstructed), dtype=np.float32)
                
                mae = np.mean(np.abs(original_np - recon_np))
                mse = np.mean((original_np - recon_np) ** 2)
                psnr = 10 * np.log10(255**2 / (mse + 1e-10))
                
                all_maes.append(mae)
                all_mses.append(mse)
                all_psnrs.append(psnr)
                
                # Save examples (first 2 per mask level)
                if len([e for e in self.examples if e["mask_frac"] == mask_frac]) < 2:
                    self.examples.append({
                        "original": img,
                        "masked": bytes(masked),
                        "reconstructed": reconstructed,
                        "mask_frac": mask_frac,
                        "mae": mae,
                        "psnr": psnr,
                        "label": test_labels[img_idx] if img_idx < len(test_labels) else -1,
                    })
            
            self.results[mask_frac] = {
                "mae": np.mean(all_maes),
                "mse": np.mean(all_mses),
                "psnr": np.mean(all_psnrs),
                "n_images": len(test_images),
            }
            
            print(f"[image_gen] Mask {mask_frac*100:.0f}%: "
                  f"MAE={np.mean(all_maes):.2f}, PSNR={np.mean(all_psnrs):.2f}dB")
        
        self.observe(state)
        print("[image_gen] Experiment complete.")
