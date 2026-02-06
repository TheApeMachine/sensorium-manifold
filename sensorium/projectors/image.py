"""Image generation projectors for tables and figures.

Generates summary tables and visualization figures for image inpainting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class ImageFigureConfig:
    """Configuration for image figure."""
    name: str = "image_gen"
    format: str = "png"
    dpi: int = 300


class ImageTableProjector(BaseProjector):
    """Projector for image generation summary table."""
    
    def __init__(
        self,
        output_dir: Path | None = None,
        name: str = "image_gen_summary",
        train_images: int = 100,
        test_images: int = 20,
    ):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
        self.train_images = train_images
        self.test_images = test_images
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        # Get mask_results from first result
        result = results[0]
        mask_results = result.get("mask_results", {})
        
        if not mask_results:
            return {"status": "skipped", "reason": "no mask results"}
        
        self.ensure_output_dir()
        
        mask_levels = [0.1, 0.2, 0.3, 0.5]
        
        table_content = r"""\begin{table}[t]
\centering
\caption{MNIST inpainting via thermodynamic trie. The manifold learns pixel patterns from training images, then reconstructs masked regions in test images using dual-domain inference. PSNR (Peak Signal-to-Noise Ratio) measures reconstruction quality; MAE (Mean Absolute Error) measures pixel-level deviation.}
\label{tab:image_gen}
\begin{tabular}{l c c c c}
\toprule
\textbf{Metric} & \textbf{10\% Mask} & \textbf{20\% Mask} & \textbf{30\% Mask} & \textbf{50\% Mask} \\
\midrule
"""
        
        psnrs = [f"{mask_results[m]['psnr']:.1f}" if m in mask_results else "---" for m in mask_levels]
        table_content += f"PSNR (dB) & {' & '.join(psnrs)} \\\\\n"
        
        maes = [f"{mask_results[m]['mae']:.1f}" if m in mask_results else "---" for m in mask_levels]
        table_content += f"MAE (pixels) & {' & '.join(maes)} \\\\\n"
        
        mses = [f"{mask_results[m]['mse']:.0f}" if m in mask_results else "---" for m in mask_levels]
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
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class ImageFigureProjector(BaseProjector):
    """Projector for image generation 3-panel figure."""
    
    def __init__(
        self,
        config: ImageFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = ImageFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 3-panel visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        result = results[0]
        mask_results = result.get("mask_results", {})
        examples = result.get("examples", [])
        
        if not mask_results:
            return {"status": "skipped", "reason": "no mask results"}
        
        self.ensure_output_dir()
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel A: Example reconstructions at 30% mask
        ax = axes[0]
        target_mask = 0.3
        examples_at_level = [e for e in examples if e.get("mask_frac") == target_mask]
        
        if len(examples_at_level) < 4:
            examples_at_level = examples[:4]
        
        n_examples = min(4, len(examples_at_level))
        
        if n_examples > 0:
            composite = np.ones((28 * 2, 28 * n_examples + n_examples - 1, 3)) * 0.9
            
            for i, example in enumerate(examples_at_level[:n_examples]):
                original = np.array(list(example["original"])).reshape(28, 28)
                recon = np.array(list(example["reconstructed"])).reshape(28, 28)
                
                x_offset = i * 29
                
                composite[:28, x_offset:x_offset+28, 0] = original / 255
                composite[:28, x_offset:x_offset+28, 1] = original / 255
                composite[:28, x_offset:x_offset+28, 2] = original / 255
                
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
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: PSNR vs mask fraction
        ax = axes[1]
        mask_fracs = sorted(mask_results.keys())
        psnrs = [mask_results[m]["psnr"] for m in mask_fracs]
        maes = [mask_results[m]["mae"] for m in mask_fracs]
        
        color1 = '#336699'
        ax.plot([m * 100 for m in mask_fracs], psnrs, 'o-', color=color1, 
               linewidth=2, markersize=8, label='PSNR')
        ax.set_xlabel("Mask fraction (%)", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10, color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_ylim(10, 25)
        
        ax2 = ax.twinx()
        color2 = '#4C994C'
        ax2.plot([m * 100 for m in mask_fracs], maes, 's--', color=color2,
                linewidth=2, markersize=8, label='MAE')
        ax2.set_ylabel("MAE (pixel intensity)", fontsize=10, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 30)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: MAE breakdown by mask level
        ax = axes[2]
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
        
        mask_fracs = sorted(mask_results.keys())
        x_pos = np.arange(len(mask_fracs))
        maes = [mask_results[m]["mae"] for m in mask_fracs]
        
        bars = ax.bar(x_pos, maes, color=colors[:len(mask_fracs)], 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{mae:.1f}", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(m*100)}%" for m in mask_fracs], fontsize=10)
        ax.set_xlabel("Mask fraction", fontsize=10)
        ax.set_ylabel("Mean Absolute Error", fontsize=10)
        ax.set_ylim(0, max(maes) * 1.2 if maes else 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
