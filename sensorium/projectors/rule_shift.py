"""Rule shift projectors for tables and figures.

Generates LaTeX tables and matplotlib figures showing
how the manifold adapts when pattern rules change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class RuleShiftFigureConfig:
    """Configuration for rule shift figure."""
    name: str = "rule_shift"
    format: str = "png"
    dpi: int = 300


class RuleShiftTableProjector(BaseProjector):
    """Custom projector for rule shift summary table."""
    
    def __init__(self, output_dir: Path | None = None, name: str = "rule_shift_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        # Get first (and likely only) result
        result = results[0]
        accuracy_history = result.get("accuracy_history", [])
        forward_reps = result.get("forward_reps", 50)
        segment_size = result.get("segment_size", 24)
        context_length = result.get("context_length", 8)
        
        if not accuracy_history:
            return {"status": "skipped", "reason": "no accuracy history"}
        
        self.ensure_output_dir()
        
        forward_accs = [r["accuracy"] for r in accuracy_history if r["phase"] == "forward"]
        reverse_accs = [r["accuracy"] for r in accuracy_history if r["phase"] == "reverse"]
        
        forward_baseline = np.mean(forward_accs) if forward_accs else 0.0
        forward_final = forward_accs[-1] if forward_accs else 0.0
        reverse_initial = reverse_accs[0] if reverse_accs else 0.0
        reverse_final = reverse_accs[-1] if reverse_accs else 0.0
        
        # Find recovery point
        recovery_rep = None
        threshold = forward_baseline * 0.8
        for r in accuracy_history:
            if r["phase"] == "reverse" and r["accuracy"] >= threshold:
                recovery_rep = r["rep"] - forward_reps
                break
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Rule-shift adaptation results. The manifold learns forward transitions, then adapts online when the sequence reverses. Recovery time measures how quickly the system regains baseline accuracy after the rule shift.}
\label{tab:rule_shift}
\begin{tabular}{l c c}
\toprule
\textbf{Metric} & \textbf{Forward Phase} & \textbf{Reverse Phase} \\
\midrule
"""
        table_content += f"Mean accuracy & {forward_baseline*100:.1f}\\% & {(np.mean(reverse_accs) if reverse_accs else 0)*100:.1f}\\% \\\\\n"
        table_content += f"Final accuracy & {forward_final*100:.1f}\\% & {reverse_final*100:.1f}\\% \\\\\n"
        table_content += f"Initial accuracy & --- & {reverse_initial*100:.1f}\\% \\\\\n"
        table_content += r"\midrule" + "\n"
        table_content += r"\multicolumn{3}{l}{\textit{Adaptation Dynamics}} \\" + "\n"
        table_content += f"\\quad Phase switch (rep) & \\multicolumn{{2}}{{c}}{{{forward_reps}}} \\\\\n"
        table_content += f"\\quad Recovery (reps after switch) & \\multicolumn{{2}}{{c}}{{{recovery_rep if recovery_rep else 'N/A'}}} \\\\\n"
        table_content += f"\\quad Segment size & \\multicolumn{{2}}{{c}}{{{segment_size}}} \\\\\n"
        table_content += f"\\quad Context length & \\multicolumn{{2}}{{c}}{{{context_length}}} \\\\\n"
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class RuleShiftFigureProjector(BaseProjector):
    """Custom projector for rule shift visualization."""
    
    def __init__(
        self,
        config: RuleShiftFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = RuleShiftFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 3-panel visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        result = results[0]
        accuracy_history = result.get("accuracy_history", [])
        forward_reps = result.get("forward_reps", 50)
        
        if not accuracy_history:
            return {"status": "skipped", "reason": "no accuracy history"}
        
        self.ensure_output_dir()
        
        reps = [r["rep"] for r in accuracy_history]
        accuracies = [r["accuracy"] for r in accuracy_history]
        phases = [r["phase"] for r in accuracy_history]
        
        forward_reps_list = [r for r, p in zip(reps, phases) if p == "forward"]
        forward_accs = [a for a, p in zip(accuracies, phases) if p == "forward"]
        reverse_reps_list = [r for r, p in zip(reps, phases) if p == "reverse"]
        reverse_accs = [a for a, p in zip(accuracies, phases) if p == "reverse"]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel A: Accuracy over time
        ax = axes[0]
        ax.plot(forward_reps_list, forward_accs, 'o-', color='#336699', 
               linewidth=2, markersize=6, label='Forward')
        ax.plot(reverse_reps_list, reverse_accs, 's-', color='#4C994C',
               linewidth=2, markersize=6, label='Reverse')
        ax.axvline(x=forward_reps, color='red', linestyle='--', linewidth=2, label='Rule Shift')
        if forward_accs:
            ax.axhline(y=np.mean(forward_accs), color='#336699', linestyle=':', alpha=0.5)
        ax.set_xlabel("Training repetition", fontsize=10)
        ax.set_ylabel("Next-byte accuracy", fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: Accuracy comparison bar chart
        ax = axes[1]
        x_pos = np.arange(4)
        values = [
            np.mean(forward_accs) if forward_accs else 0,
            forward_accs[-1] if forward_accs else 0,
            reverse_accs[0] if reverse_accs else 0,
            reverse_accs[-1] if reverse_accs else 0,
        ]
        labels = ["Fwd Mean", "Fwd Final", "Rev Initial", "Rev Final"]
        colors = ['#336699', '#336699', '#4C994C', '#4C994C']
        
        bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.0%}", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Adaptation rate
        ax = axes[2]
        if reverse_accs and len(reverse_accs) > 1:
            initial = reverse_accs[0]
            deltas = [acc - initial for acc in reverse_accs]
            delta_reps = [r - forward_reps for r in reverse_reps_list]
            
            ax.fill_between(delta_reps, 0, deltas, alpha=0.3, color='#4C994C')
            ax.plot(delta_reps, deltas, 'o-', color='#4C994C', linewidth=2, markersize=6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel("Reps after rule shift", fontsize=10)
            ax.set_ylabel("Accuracy gain from initial", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
