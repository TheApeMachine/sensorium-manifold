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
        
        # Generate formatted LaTeX table
        self._generate_table()
        
        # Generate 3-panel figure
        self._generate_figure()
        
        print(f"✓ Generated: paper/tables/image_gen_summary.tex")
    
    def _generate_table(self):
        """Generate properly formatted LaTeX table."""
        
        table_content = r"""\begin{table}[t]
\centering
\caption{MNIST inpainting via thermodynamic trie. The manifold learns pixel patterns from training images, then reconstructs masked regions in test images using dual-domain inference. PSNR (Peak Signal-to-Noise Ratio) measures reconstruction quality; MAE (Mean Absolute Error) measures pixel-level deviation.}
\label{tab:image_gen}
\begin{tabular}{l c c c c}
\toprule
\textbf{Metric} & \textbf{10\% Mask} & \textbf{20\% Mask} & \textbf{30\% Mask} & \textbf{50\% Mask} \\
\midrule
"""
        
        # Get results for each mask level
        mask_levels = [0.1, 0.2, 0.3, 0.5]
        
        # PSNR row
        psnrs = [f"{self.results[m]['psnr']:.1f}" if m in self.results else "---" for m in mask_levels]
        table_content += f"PSNR (dB) & {' & '.join(psnrs)} \\\\\n"
        
        # MAE row  
        maes = [f"{self.results[m]['mae']:.1f}" if m in self.results else "---" for m in mask_levels]
        table_content += f"MAE (pixels) & {' & '.join(maes)} \\\\\n"
        
        # MSE row
        mses = [f"{self.results[m]['mse']:.0f}" if m in self.results else "---" for m in mask_levels]
        table_content += f"MSE & {' & '.join(mses)} \\\\\n"
        
        table_content += r"""\midrule
\multicolumn{5}{l}{\textit{Dataset}} \\
\quad Training images & \multicolumn{4}{c}{""" + str(self.train_images) + r"""} \\
\quad Test images & \multicolumn{4}{c}{""" + str(self.test_images) + r"""} \\
\quad Image size & \multicolumn{4}{c}{28$\times$28 = 784 pixels} \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "image_gen_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
    
    def _generate_figure(self):
        """Generate 3-panel visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # =================================================================
        # Panel A: Example reconstructions - different digits at 30% mask
        # =================================================================
        ax = axes[0]
        
        # Show 4 different digits at the same mask level (30%)
        # This demonstrates the system works across different digit types
        target_mask = 0.3
        examples_at_level = [e for e in self.examples if e["mask_frac"] == target_mask]
        
        # If not enough at 30%, use all examples
        if len(examples_at_level) < 4:
            examples_at_level = self.examples[:4]
        
        n_examples = min(4, len(examples_at_level))
        
        # Create composite image: 2 rows x n_examples columns
        composite = np.ones((28 * 2, 28 * n_examples + n_examples - 1, 3)) * 0.9
        
        for i, example in enumerate(examples_at_level[:n_examples]):
            original = np.array(list(example["original"])).reshape(28, 28)
            recon = np.array(list(example["reconstructed"])).reshape(28, 28)
            
            x_offset = i * 29
            
            # Top row: original
            composite[:28, x_offset:x_offset+28, 0] = original / 255
            composite[:28, x_offset:x_offset+28, 1] = original / 255
            composite[:28, x_offset:x_offset+28, 2] = original / 255
            
            # Bottom row: reconstructed
            composite[28:56, x_offset:x_offset+28, 0] = recon / 255
            composite[28:56, x_offset:x_offset+28, 1] = recon / 255
            composite[28:56, x_offset:x_offset+28, 2] = recon / 255
        
        ax.imshow(composite)
        ax.set_xticks([14 + i * 29 for i in range(n_examples)])
        labels = [f"d={examples_at_level[i].get('label', '?')}" for i in range(n_examples)]
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel(f"Digit (at {int(target_mask*100)}% mask)", fontsize=10)
        ax.set_yticks([14, 42])
        ax.set_yticklabels(["Original", "Reconstructed"], fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: PSNR vs mask fraction
        # =================================================================
        ax = axes[1]
        
        mask_fracs = sorted(self.results.keys())
        psnrs = [self.results[m]["psnr"] for m in mask_fracs]
        maes = [self.results[m]["mae"] for m in mask_fracs]
        
        color1 = '#336699'
        ax.plot([m * 100 for m in mask_fracs], psnrs, 'o-', color=color1, 
               linewidth=2, markersize=8, label='PSNR')
        ax.set_xlabel("Mask fraction (%)", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10, color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_ylim(10, 25)
        
        # Secondary axis for MAE
        ax2 = ax.twinx()
        color2 = '#4C994C'
        ax2.plot([m * 100 for m in mask_fracs], maes, 's--', color=color2,
                linewidth=2, markersize=8, label='MAE')
        ax2.set_ylabel("MAE (pixel intensity)", fontsize=10, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 30)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: MAE breakdown by mask level (bar chart)
        # =================================================================
        ax = axes[2]
        
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
        
        mask_fracs = sorted(self.results.keys())
        x_pos = np.arange(len(mask_fracs))
        
        # Show MAE as bars
        maes = [self.results[m]["mae"] for m in mask_fracs]
        bars = ax.bar(x_pos, maes, color=colors[:len(mask_fracs)], 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{mae:.1f}", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(m*100)}%" for m in mask_fracs], fontsize=10)
        ax.set_xlabel("Mask fraction", fontsize=10)
        ax.set_ylabel("Mean Absolute Error", fontsize=10)
        ax.set_ylim(0, max(maes) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "image_gen.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")

    def run(self):
        """Run MNIST inpainting experiment."""
        import time
        
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
        grid_size = (32, 32, 32)
        dt = 0.01
        
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
                    grid_size=grid_size,
                    dt=dt,
                ),
                spectral=SpectralSimulationConfig(
                    grid_size=grid_size,
                    dt=dt,
                ),
                tokenizer=tokenizer_config,
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "spectral": InferenceObserver([CarrierObserver(None)])
            }
        )
        
        start_time = time.time()
        state = manifold.run()
        wall_time_ms = (time.time() - start_time) * 1000
        
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
                
                # Save examples (up to 5 per mask level to show variety)
                if len([e for e in self.examples if e["mask_frac"] == mask_frac]) < 5:
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
        
        # Get carrier stats
        carriers = manifold.carriers or {}
        amplitudes = carriers.get("amplitudes")
        n_carriers = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
        crystallized = carriers.get("crystallized")
        n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
        n_particles = len(geo_state["token_ids"]) if geo_state.get("token_ids") is not None else 0
        
        self.observe(state)
        
        # Write simulation stats
        self.write_simulation_stats(
            "image_gen",
            n_particles=n_particles,
            n_carriers=n_carriers,
            n_crystallized=n_crystallized,
            grid_size=grid_size,
            dt=dt,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/image_gen_stats.tex")
        
        print("[image_gen] Experiment complete.")
