"""Text diffusion experiment projectors.

Custom projectors for text diffusion visualization and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING

import numpy as np

from sensorium.projectors.base import BaseProjector

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class DiffusionFigureConfig:
    """Configuration for diffusion figure.
    
    Attributes:
        name: Output filename (without extension)
        format: Output format (png, pdf, svg)
        dpi: Output DPI
    """
    name: str = "text_diffusion"
    format: str = "png"
    dpi: int = 300


class DiffusionTableProjector(BaseProjector):
    """Projector for text diffusion summary table.
    
    Generates a LaTeX table showing accuracy at different mask levels.
    
    Expects InferenceObserver results with fields:
    - mask_frac: Mask fraction (0.1, 0.2, etc.)
    - char_accuracy: Character-level accuracy
    - n_correct: Number of correct predictions
    - n_masked: Number of masked positions
    - hamming_dist: Hamming distance
    """
    
    def __init__(self, output_dir: Path | None = None, name: str = "text_diffusion_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        # Get results list from source
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        self.ensure_output_dir()
        
        # Sort by mask_frac
        results = sorted(results, key=lambda r: r.get("mask_frac", 0))
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Text byte denoising via thermodynamic trie. Masked characters are reconstructed using pattern matching from training data. Accuracy measures exact character recovery at masked positions.}
\label{tab:text_diffusion}
\begin{tabular}{l c c c}
\toprule
\textbf{Mask Level} & \textbf{Accuracy} & \textbf{Correct/Masked} & \textbf{Hamming Dist.} \\
\midrule
"""
        for res in results:
            mask_frac = res.get("mask_frac", 0)
            char_accuracy = res.get("char_accuracy", 0)
            n_correct = res.get("n_correct", 0)
            n_masked = res.get("n_masked", 0)
            hamming = res.get("hamming_dist", 0)
            
            table_content += f"{mask_frac*100:.0f}\\% & "
            table_content += f"{char_accuracy*100:.1f}\\% & "
            table_content += f"{n_correct}/{n_masked} & "
            table_content += f"{hamming} \\\\\n"
        
        # Average accuracy
        accuracies = [r.get("char_accuracy", 0) for r in results]
        avg_acc = np.mean(accuracies) if accuracies else 0
        
        table_content += r"\midrule" + "\n"
        table_content += f"\\textbf{{Average}} & {avg_acc*100:.1f}\\% & --- & --- \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class DiffusionFigureProjector(BaseProjector):
    """Projector for text diffusion 2-panel visualization.
    
    Generates:
    A) Line plot of accuracy vs mask level
    B) Bar chart of accuracy at each mask level
    
    Expects InferenceObserver results with fields:
    - mask_frac: Mask fraction
    - char_accuracy: Character-level accuracy
    """
    
    def __init__(
        self,
        config: DiffusionFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        
        if config:
            self.config = config
        else:
            self.config = DiffusionFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 2-panel visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Get results list from source
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        self.ensure_output_dir()
        
        # Sort by mask_frac and extract data
        results = sorted(results, key=lambda r: r.get("mask_frac", 0))
        mask_fracs = [r.get("mask_frac", 0) for r in results]
        accuracies = [r.get("char_accuracy", 0) for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Panel A: Accuracy vs mask level (line plot)
        ax = axes[0]
        ax.plot([mf * 100 for mf in mask_fracs], accuracies, 'o-',
               color='#336699', linewidth=2, markersize=10)
        ax.axhline(y=1/256, color='red', linestyle='--', alpha=0.5,
                  label='Random (1/256)')
        ax.set_xlabel("Mask percentage", fontsize=11)
        ax.set_ylabel("Character accuracy", fontsize=11)
        max_acc = max(accuracies) if accuracies else 0.2
        ax.set_ylim(0, max_acc * 1.2 if max_acc > 0 else 0.2)
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        # Panel B: Accuracy bar chart
        ax = axes[1]
        x_pos = np.arange(len(mask_fracs))
        colors = plt.cm.viridis([a / max_acc if max_acc > 0 else 0 for a in accuracies])
        bars = ax.bar(x_pos, accuracies, width=0.6, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.9)
        
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.1%}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.axhline(y=1/256, color='red', linestyle='--', alpha=0.7,
                  label=f'Random baseline ({1/256:.2%})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{mf*100:.0f}%" for mf in mask_fracs], fontsize=10)
        ax.set_xlabel("Mask level", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, max_acc * 1.3 if max_acc > 0 else 0.2)
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
