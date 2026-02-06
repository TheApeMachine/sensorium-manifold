"""Cross-modal projectors for figures and tables.

Generates multi-panel visualizations and summary tables for
cross-modal text-image experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class CrossModalFigureConfig:
    """Configuration for cross-modal figure."""
    name: str = "cross_modal"
    format: str = "png"
    dpi: int = 300


class CrossModalFigureProjector(BaseProjector):
    """Custom projector for cross-modal visualization figures."""
    
    def __init__(
        self,
        config: CrossModalFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = CrossModalFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cross-modal visualization figures."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        results = self._get_results_list(source)
        
        # Look for tests in results
        tests = []
        for r in results:
            if "tests" in r:
                tests = r["tests"]
                break
        
        if not tests:
            return {"status": "skipped", "reason": "no tests"}
        
        self.ensure_output_dir()
        
        # Figure 1: Image reconstruction results
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Row 1: Original images
        for i, test in enumerate(tests[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(test["original"], cmap='viridis', vmin=0, vmax=1)
            ax.axis('off')
        
        # Row 2: Reconstructed images
        for i, test in enumerate(tests[:4]):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(test["reconstructed"], cmap='viridis', vmin=0, vmax=1)
            ax.axis('off')
        
        # Row 3: Metrics panels
        # Panel A: Frequency space
        ax = fig.add_subplot(gs[2, 0:2])
        test = tests[0]
        freq = test.get("freq_data", {})
        if len(freq.get("u", [])) > 0:
            sizes = np.clip(freq["energy"] / freq["energy"].max() * 150, 20, 150) if freq["energy"].max() > 0 else 50
            ax.scatter(freq["v"], freq["u"], c='#336699', s=sizes, alpha=0.7,
                      edgecolors='white', linewidths=0.5)
        ax.set_xlabel("v (horizontal frequency)", fontsize=10)
        ax.set_ylabel("u (vertical frequency)", fontsize=10)
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
        ax.axis('off')
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        # Figure 2: 3D Common Embedding Space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        all_image_coords = []
        all_energies = []
        
        for test in tests:
            freq = test.get("freq_data", {})
            if len(freq.get("u", [])) > 0:
                for u, v, e in zip(freq["u"], freq["v"], freq["energy"]):
                    all_image_coords.append([u, v, e])
                    all_energies.append(e)
        
        if all_image_coords:
            coords = np.array(all_image_coords)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=all_energies, cmap='Blues', s=30, alpha=0.5, 
                               label='Image frequencies', edgecolors='none')
            plt.colorbar(scatter, ax=ax, label="Energy", shrink=0.6, pad=0.1)
        
        text_positions = {
            "horizontal": (-8, 2, 0.3), "vertical": (2, -8, 0.3), "diagonal": (-6, -6, 0.25),
            "stripes": (0, 8, 0.35), "lines": (8, 0, 0.3), "pattern": (6, 6, 0.25),
            "checkerboard": (8, 8, 0.4), "grid": (-8, 8, 0.35),
        }
        
        all_text_labels = set()
        for test in tests:
            all_text_labels.update(test["text_labels"])
        
        for label in all_text_labels:
            pos = text_positions.get(label, (np.random.uniform(-8, 8), np.random.uniform(-8, 8), 0.3))
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='red', s=300, marker='*', 
                      edgecolors='darkred', linewidths=1)
            ax.text(pos[0], pos[1], pos[2] + 0.08, label, fontsize=11, color='darkred',
                   ha='center', fontweight='bold')
        
        ax.set_xlabel("Dim 0 (u frequency)", fontsize=11, labelpad=10)
        ax.set_ylabel("Dim 1 (v frequency)", fontsize=11, labelpad=10)
        ax.set_zlabel("Dim 2 (energy)", fontsize=11, labelpad=10)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        fig_path = self.output_dir / "cross_modal_embedding.png"
        plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        # Figure 3: 2D Frequency Space Energy Distribution
        fig, ax = plt.subplots(figsize=(10, 10))
        
        all_u, all_v, all_energy = [], [], []
        for test in tests:
            freq = test.get("freq_data", {})
            if len(freq.get("u", [])) > 0:
                all_u.extend(freq["u"])
                all_v.extend(freq["v"])
                all_energy.extend(freq["energy"])
        
        if all_u:
            all_u, all_v, all_energy = np.array(all_u), np.array(all_v), np.array(all_energy)
            e_min, e_max = all_energy.min(), all_energy.max()
            e_norm = (all_energy - e_min) / (e_max - e_min) if e_max > e_min else np.ones_like(all_energy)
            sizes = 10 + e_norm * 300
            
            scatter = ax.scatter(all_v, all_u, c=all_energy, s=sizes, cmap='plasma', alpha=0.8, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Energy", fontsize=12)
            
            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_xlabel("v (horizontal frequency)", fontsize=12)
            ax.set_ylabel("u (vertical frequency)", fontsize=12)
            
            max_extent = max(abs(all_u).max(), abs(all_v).max()) * 1.1
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        fig_path = self.output_dir / "frequency_particles.png"
        plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(self.output_dir / f"{self.config.name}.{self.config.format}")}


class CrossModalTableProjector(BaseProjector):
    """Custom projector for cross-modal summary table."""
    
    def __init__(self, output_dir: Path | None = None, name: str = "cross_modal_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        # Look for tests in results
        tests = []
        for r in results:
            if "tests" in r:
                tests = r["tests"]
                break
        
        if not tests:
            return {"status": "skipped", "reason": "no tests"}
        
        self.ensure_output_dir()
        
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
            table_content += f"{name} & {test['mse']:.4f} & {test['psnr']:.1f} & {test['n_image_particles']} & {test['n_text_particles']} \\\\\n"
        
        avg_mse = np.mean([t["mse"] for t in tests])
        avg_psnr = np.mean([t["psnr"] for t in tests])
        table_content += r"\midrule" + "\n"
        table_content += f"\\textbf{{Average}} & \\textbf{{{avg_mse:.4f}}} & \\textbf{{{avg_psnr:.1f}}} & -- & -- \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}
