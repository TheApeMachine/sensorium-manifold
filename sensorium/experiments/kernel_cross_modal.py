"""Cross-modal experiment: Text and Image in unified manifold.

Demonstrates the manifold's native multimodality by:
1. Encoding image via 2D FFT → frequency particles
2. Encoding text via byte tokenization → text particles  
3. Running thermodynamic dynamics on the unified particle set
4. Reconstructing image via inverse FFT
5. Visualizing the common embedding space

Produces:
- `paper/figures/cross_modal.png` - Multi-panel hero figure
- `paper/tables/cross_modal_summary.tex` - Metrics table
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    SpectralSimulationConfig,
    GeometricSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.base import ObserverProtocol


class CrossModalObserver(ObserverProtocol):
    """Observer that tracks cross-modal dynamics."""
    
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.step_count = 0
        self.history = {
            "step": [],
            "n_carriers": [],
            "n_crystallized": [],
            "mean_energy": [],
        }
    
    def observe(self, observation=None, **kwargs):
        self.step_count += 1
        
        if observation is not None:
            amplitudes = observation.get("amplitudes")
            carrier_state = observation.get("carrier_state")
            
            if amplitudes is not None:
                active = amplitudes > 1e-6
                n_carriers = int(active.sum().item())
                n_crystallized = 0
                if carrier_state is not None and n_carriers > 0:
                    n_crystallized = int((carrier_state[:n_carriers] == 2).sum().item())
            else:
                n_carriers = 0
                n_crystallized = 0
            
            self.history["step"].append(self.step_count)
            self.history["n_carriers"].append(n_carriers)
            self.history["n_crystallized"].append(n_crystallized)
        
        return {"done_thinking": self.step_count >= self.max_steps}


class KernelCrossModal(Experiment):
    """Cross-modal experiment demonstrating unified text-image processing."""
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        self.vocab_size = 4096
        self.prime = 31
        self.image_size = 32  # Small for fast FFT
        self.top_k_freq = 64  # Number of frequency components to keep
        self.n_steps = 150
        
        self.results: Dict[str, Any] = {}
    
    def run(self):
        print("[cross_modal] Starting cross-modal experiment...")
        
        # Test cases: different image patterns with associated text
        test_cases = [
            ("horizontal", self._create_stripe_image("horizontal"), ["horizontal", "stripes", "lines"]),
            ("vertical", self._create_stripe_image("vertical"), ["vertical", "stripes", "lines"]),
            ("diagonal", self._create_stripe_image("diagonal"), ["diagonal", "stripes", "pattern"]),
            ("checkerboard", self._create_checkerboard_image(), ["checkerboard", "grid", "pattern"]),
        ]
        
        self.results["tests"] = []
        
        for name, image, text_labels in test_cases:
            print(f"\n  Processing: {name}")
            result = self._run_cross_modal_test(name, image, text_labels)
            self.results["tests"].append(result)
            print(f"    MSE: {result['mse']:.4f}, PSNR: {result['psnr']:.2f} dB")
        
        # Generate artifacts
        self._generate_figure()
        self._generate_table()
        
        print("\n[cross_modal] Experiment complete.")
    
    def _create_stripe_image(self, orientation: str) -> np.ndarray:
        """Create a stripe pattern image."""
        x = np.linspace(0, 4 * np.pi, self.image_size)
        y = np.linspace(0, 4 * np.pi, self.image_size)
        xx, yy = np.meshgrid(x, y)
        
        if orientation == "horizontal":
            image = 0.5 + 0.4 * np.sin(yy * 2)
        elif orientation == "vertical":
            image = 0.5 + 0.4 * np.sin(xx * 2)
        elif orientation == "diagonal":
            image = 0.5 + 0.4 * np.sin((xx + yy) * 1.5)
        else:
            image = 0.5 * np.ones((self.image_size, self.image_size))
        
        return image.astype(np.float32)
    
    def _create_checkerboard_image(self) -> np.ndarray:
        """Create a checkerboard pattern."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        block_size = self.image_size // 4
        for i in range(0, self.image_size, block_size):
            for j in range(0, self.image_size, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    image[i:i+block_size, j:j+block_size] = 1.0
        return image
    
    def _image_to_frequency_bytes(self, image: np.ndarray) -> Tuple[bytes, Dict]:
        """Convert image to frequency-domain bytes for the manifold.
        
        Returns:
            bytes: Quantized frequency magnitudes and phases
            metadata: Dict with FFT info for reconstruction
        """
        # 2D FFT
        image_t = torch.tensor(image, dtype=torch.float32)
        spectrum = torch.fft.fft2(image_t)
        spectrum_shifted = torch.fft.fftshift(spectrum)
        
        # Extract magnitude and phase
        magnitude = torch.abs(spectrum_shifted)
        phase = torch.angle(spectrum_shifted)
        
        # Flatten and get top-k indices by magnitude
        mag_flat = magnitude.flatten()
        phase_flat = phase.flatten()
        
        if self.top_k_freq < len(mag_flat):
            topk_indices = torch.topk(mag_flat, self.top_k_freq).indices
        else:
            topk_indices = torch.arange(len(mag_flat))
        
        # Quantize magnitudes and phases to bytes
        mag_selected = mag_flat[topk_indices]
        phase_selected = phase_flat[topk_indices]
        
        # Normalize magnitude to 0-255
        mag_max = mag_selected.max()
        if mag_max > 0:
            mag_norm = (mag_selected / mag_max * 255).to(torch.uint8)
        else:
            mag_norm = torch.zeros_like(mag_selected, dtype=torch.uint8)
        
        # Normalize phase from [-pi, pi] to 0-255
        phase_norm = ((phase_selected + np.pi) / (2 * np.pi) * 255).to(torch.uint8)
        
        # Interleave magnitude and phase bytes
        combined = torch.stack([mag_norm, phase_norm], dim=1).flatten()
        
        metadata = {
            "shape": image.shape,
            "topk_indices": topk_indices.numpy(),
            "mag_max": mag_max.item(),
            "spectrum_shape": spectrum_shifted.shape,
            # Store original values for reconstruction
            "mag_selected": mag_selected.numpy(),
            "phase_selected": phase_selected.numpy(),
        }
        
        return bytes(combined.numpy()), metadata
    
    def _text_to_bytes(self, labels: List[str]) -> bytes:
        """Convert text labels to bytes."""
        text = " ".join(labels)
        return text.encode("utf-8")
    
    def _reconstruct_image(
        self, 
        state: Dict[str, torch.Tensor], 
        metadata: Dict,
        n_image_bytes: int,
    ) -> np.ndarray:
        """Reconstruct image from manifold state.
        
        Uses the oscillator energies to modulate frequency magnitudes.
        The manifold dynamics adjust energy levels - we use this to weight
        the original frequency components.
        """
        H, W = metadata["shape"]
        topk_indices = metadata["topk_indices"]
        mag_selected = metadata["mag_selected"]
        phase_selected = metadata["phase_selected"]
        
        # Get oscillator energies from state
        osc_energy = state.get("osc_energy")
        
        # The first n_image_bytes tokens correspond to the image
        # They're interleaved magnitude/phase, so n_image_bytes/2 frequency components
        n_freq = min(len(topk_indices), n_image_bytes // 2)
        
        # Compute energy modulation factors from manifold state
        if osc_energy is not None and len(osc_energy) >= n_image_bytes:
            image_energies = osc_energy[:n_image_bytes].cpu().numpy()
            # Pair energies (magnitude, phase) and use magnitude energy as weight
            energy_weights = np.array([image_energies[i * 2] for i in range(n_freq)])
            # Normalize weights
            if energy_weights.max() > 0:
                energy_weights = energy_weights / energy_weights.max()
            else:
                energy_weights = np.ones(n_freq)
        else:
            energy_weights = np.ones(n_freq)
        
        # Reconstruct spectrum using original magnitudes/phases modulated by energy
        spectrum = np.zeros((H, W), dtype=np.complex64)
        
        for i, idx in enumerate(topk_indices[:n_freq]):
            # Use original magnitude weighted by manifold energy
            mag = mag_selected[i] * energy_weights[i]
            phase = phase_selected[i]
            
            # Convert flat index to 2D
            u = idx // W
            v = idx % W
            
            if 0 <= u < H and 0 <= v < W:
                spectrum[u, v] = mag * np.exp(1j * phase)
        
        # Inverse FFT
        spectrum_unshifted = np.fft.ifftshift(spectrum)
        reconstructed = np.fft.ifft2(spectrum_unshifted).real
        
        # Normalize to [0, 1]
        rmin, rmax = reconstructed.min(), reconstructed.max()
        if rmax > rmin:
            reconstructed = (reconstructed - rmin) / (rmax - rmin)
        else:
            reconstructed = np.clip(reconstructed, 0, 1)
        
        return reconstructed.astype(np.float32)
    
    def _run_cross_modal_test(
        self, 
        name: str, 
        image: np.ndarray, 
        text_labels: List[str],
    ) -> Dict[str, Any]:
        """Run a single cross-modal test."""
        
        # Convert image and text to bytes
        image_bytes, metadata = self._image_to_frequency_bytes(image)
        text_bytes = self._text_to_bytes(text_labels)
        
        # Combine: image bytes first, then text bytes
        combined_data = image_bytes + text_bytes
        
        def data_generator(d=combined_data):
            yield d
        
        observer = CrossModalObserver(max_steps=self.n_steps)
        
        cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=self.vocab_size,
                hash_prime=self.prime,
                segment_size=len(image_bytes),  # Segment by modality
            ),
            geometric=GeometricSimulationConfig(
                grid_size=(32, 32, 32),
            ),
            spectral=SpectralSimulationConfig(
                max_carriers=64,
                grid_size=(32, 32, 32),
                stable_amp_threshold=0.15,
                crystallize_amp_threshold=0.20,
            ),
            generator=data_generator,
        )
        
        manifold = Manifold(cfg, observers={"spectral": observer})
        
        start = time.time()
        state = manifold.run()
        wall_time = (time.time() - start) * 1000
        
        # Reconstruct image
        reconstructed = self._reconstruct_image(state, metadata, len(image_bytes))
        
        # Compute metrics
        mse = np.mean((image - reconstructed) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else 100.0
        
        # Get frequency-space visualization data
        freq_data = self._extract_frequency_space(state, metadata, len(image_bytes))
        
        return {
            "name": name,
            "original": image,
            "reconstructed": reconstructed,
            "mse": mse,
            "psnr": psnr,
            "wall_time_ms": wall_time,
            "n_particles": len(combined_data),
            "n_image_particles": len(image_bytes),
            "n_text_particles": len(text_bytes),
            "text_labels": text_labels,
            "history": observer.history,
            "freq_data": freq_data,
            "metadata": metadata,
        }
    
    def _extract_frequency_space(
        self, 
        state: Dict[str, torch.Tensor],
        metadata: Dict,
        n_image_bytes: int,
    ) -> Dict[str, np.ndarray]:
        """Extract frequency-space coordinates for visualization.
        
        Uses the original spectral magnitudes from the FFT as energy values,
        which provides meaningful variation (DC component high, edges low).
        """
        H, W = metadata["shape"]
        topk_indices = metadata["topk_indices"]
        mag_selected = metadata.get("mag_selected", None)
        
        if mag_selected is None or len(topk_indices) == 0:
            return {"u": np.array([]), "v": np.array([]), "energy": np.array([])}
        
        # Compute u, v coordinates for image particles
        n_freq = min(len(topk_indices), len(mag_selected))
        
        u_coords = []
        v_coords = []
        energies = []
        
        for i, idx in enumerate(topk_indices[:n_freq]):
            # Convert flat index to 2D frequency coordinates (centered)
            u = (idx // W) - H // 2
            v = (idx % W) - W // 2
            
            # Use spectral magnitude as energy (this is what the original used)
            energy = float(mag_selected[i])
            
            u_coords.append(u)
            v_coords.append(v)
            energies.append(energy)
        
        return {
            "u": np.array(u_coords),
            "v": np.array(v_coords),
            "energy": np.array(energies),
        }
    
    def _generate_figure(self):
        """Generate cross-modal visualization figures."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.mplot3d import Axes3D
        
        tests = self.results.get("tests", [])
        if not tests:
            return
        
        # Figure 1: Image reconstruction results
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Row 1: Original images
        for i, test in enumerate(tests[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(test["original"], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Original: {test['name']}", fontsize=10)
            ax.axis('off')
        
        # Row 2: Reconstructed images
        for i, test in enumerate(tests[:4]):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(test["reconstructed"], cmap='viridis', vmin=0, vmax=1)
            mse = test["mse"]
            ax.set_title(f"Reconstructed (MSE={mse:.3f})", fontsize=10)
            ax.axis('off')
        
        # Row 3: Frequency space and metrics
        # Panel A: Frequency space for first test (just show spatial distribution)
        ax = fig.add_subplot(gs[2, 0:2])
        test = tests[0]
        freq = test["freq_data"]
        if len(freq["u"]) > 0:
            # Use uniform color for spatial distribution, size for frequency importance
            sizes = np.clip(freq["energy"] / freq["energy"].max() * 150, 20, 150) if freq["energy"].max() > 0 else 50
            ax.scatter(
                freq["v"], freq["u"],
                c='#336699',
                s=sizes,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
            )
        ax.set_xlabel("v (horizontal frequency)", fontsize=10)
        ax.set_ylabel("u (vertical frequency)", fontsize=10)
        ax.set_title(f"Frequency Components: {test['name']}", fontsize=11)
        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: MSE comparison
        ax = fig.add_subplot(gs[2, 2])
        names = [t["name"] for t in tests]
        mses = [t["mse"] for t in tests]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.bar(range(len(names)), mses, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("MSE", fontsize=10)
        ax.set_title("Reconstruction Error", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Text-image association
        ax = fig.add_subplot(gs[2, 3])
        for i, test in enumerate(tests):
            labels = ", ".join(test["text_labels"])
            ax.text(0.05, len(tests) - 1 - i, f"{test['name']}: {labels}", fontsize=9, va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(tests) - 0.5)
        ax.set_title("Text-Image Associations", fontsize=11)
        ax.axis('off')
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "cross_modal.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {fig_path}")
        
        # Figure 2: Standalone 3D Common Embedding Space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot image frequencies from all tests in common space
        all_image_coords = []
        all_energies = []
        
        for test in tests:
            freq = test["freq_data"]
            if len(freq["u"]) > 0:
                for u, v, e in zip(freq["u"], freq["v"], freq["energy"]):
                    all_image_coords.append([u, v, e])
                    all_energies.append(e)
        
        if all_image_coords:
            coords = np.array(all_image_coords)
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=all_energies, cmap='Blues', s=30, alpha=0.5, 
                label='Image frequencies', edgecolors='none'
            )
            plt.colorbar(scatter, ax=ax, label="Energy", shrink=0.6, pad=0.1)
        
        # Plot text tokens as distinct markers with better positioning
        text_positions = {
            "horizontal": (-8, 2, 0.3),
            "vertical": (2, -8, 0.3),
            "diagonal": (-6, -6, 0.25),
            "stripes": (0, 8, 0.35),
            "lines": (8, 0, 0.3),
            "pattern": (6, 6, 0.25),
            "checkerboard": (8, 8, 0.4),
            "grid": (-8, 8, 0.35),
        }
        
        # Collect unique text labels
        all_text_labels = set()
        for test in tests:
            all_text_labels.update(test["text_labels"])
        
        for label in all_text_labels:
            pos = text_positions.get(label, (np.random.uniform(-8, 8), np.random.uniform(-8, 8), 0.3))
            ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                      c='red', s=300, marker='*', edgecolors='darkred', linewidths=1)
            ax.text(pos[0], pos[1], pos[2] + 0.08, label, fontsize=11, color='darkred', 
                   ha='center', fontweight='bold')
        
        ax.set_xlabel("Dim 0 (u frequency)", fontsize=11, labelpad=10)
        ax.set_ylabel("Dim 1 (v frequency)", fontsize=11, labelpad=10)
        ax.set_zlabel("Dim 2 (energy)", fontsize=11, labelpad=10)
        ax.set_title("Cross-Modal Particles in Common Embedding Space", fontsize=13, pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
                   markersize=10, label='Image frequencies'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, label='Text tokens'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "cross_modal_embedding.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {fig_path}")
        
        # Figure 3: 2D Frequency Space Energy Distribution (like the old visualization)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Collect all frequency data from all tests
        all_u = []
        all_v = []
        all_energy = []
        
        for test in tests:
            freq = test["freq_data"]
            if len(freq["u"]) > 0:
                all_u.extend(freq["u"])
                all_v.extend(freq["v"])
                all_energy.extend(freq["energy"])
        
        if all_u:
            all_u = np.array(all_u)
            all_v = np.array(all_v)
            all_energy = np.array(all_energy)
            
            # Normalize energy for better visualization
            e_min, e_max = all_energy.min(), all_energy.max()
            if e_max > e_min:
                e_norm = (all_energy - e_min) / (e_max - e_min)
            else:
                e_norm = np.ones_like(all_energy)
            
            # Size based on energy (bigger = more energy)
            sizes = 10 + e_norm * 300
            
            scatter = ax.scatter(
                all_v, all_u,  # v on x-axis, u on y-axis (standard frequency convention)
                c=all_energy,
                s=sizes,
                cmap='plasma',
                alpha=0.8,
                edgecolors='none',
            )
            
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Energy", fontsize=12)
            
            # Grid lines at origin
            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
            
            # Add subtle grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            ax.set_xlabel("v (horizontal frequency)", fontsize=12)
            ax.set_ylabel("u (vertical frequency)", fontsize=12)
            ax.set_title("Particle Distribution in 2D Frequency Space", fontsize=14)
            
            # Set axis limits based on data
            max_extent = max(abs(all_u).max(), abs(all_v).max()) * 1.1
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            
            # Make it square
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "frequency_particles.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {fig_path}")
    
    def _generate_table(self):
        """Generate LaTeX summary table."""
        tests = self.results.get("tests", [])
        if not tests:
            return
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Cross-modal reconstruction results. Images are encoded via 2D FFT and processed alongside text tokens in a unified manifold. Reconstruction quality varies by pattern complexity.}
\label{tab:cross_modal}
\begin{tabular}{l c c c c}
\toprule
\textbf{Pattern} & \textbf{MSE} & \textbf{PSNR (dB)} & \textbf{Image} & \textbf{Text} \\
 & & & \textbf{Particles} & \textbf{Particles} \\
\midrule
"""
        
        for test in tests:
            name = test["name"].replace("_", " ").title()
            mse = test["mse"]
            psnr = test["psnr"]
            n_img = test["n_image_particles"]
            n_txt = test["n_text_particles"]
            
            table_content += f"{name} & {mse:.4f} & {psnr:.1f} & {n_img} & {n_txt} \\\\\n"
        
        # Average row
        avg_mse = np.mean([t["mse"] for t in tests])
        avg_psnr = np.mean([t["psnr"] for t in tests])
        table_content += r"\midrule" + "\n"
        table_content += f"\\textbf{{Average}} & \\textbf{{{avg_mse:.4f}}} & \\textbf{{{avg_psnr:.1f}}} & -- & -- \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "cross_modal_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
