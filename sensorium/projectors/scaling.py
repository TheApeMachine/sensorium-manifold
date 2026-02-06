"""Scaling projectors for tables and figures.

Generates summary tables and multi-panel figures for scaling analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union, List

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class ScalingFigureConfig:
    """Configuration for scaling figures."""
    name: str = "scaling"
    format: str = "png"
    dpi: int = 300


class ScalingTableProjector(BaseProjector):
    """Projector for scaling summary table."""
    
    def __init__(self, output_dir: Path | None = None, name: str = "scaling_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        # Get data from first result
        result = results[0]
        pop = result.get("population", {})
        compute = result.get("compute", {})
        latency = result.get("latency", [])
        gen = result.get("generalization", [])
        
        self.ensure_output_dir()
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Scaling analysis summary. Carrier population dynamics show the manifold's ``carrying capacity'' and metabolic pruning behavior. Interference results show crystallization efficiency as pattern count increases.}
\label{tab:scaling}
\begin{tabular}{l r}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Population Dynamics}} \\
\quad Final carriers & """ + str(pop.get("n_carriers_final", 0)) + r""" \\
\quad Crystallized & """ + str(pop.get("n_crystallized", 0)) + r""" \\
\quad Total births & """ + str(pop.get("total_births", 0)) + r""" \\
\quad Total deaths & """ + str(pop.get("total_deaths", 0)) + r""" \\
\quad Pruning rate & """ + f"{pop.get('pruning_rate', 0):.2f}" + r""" \\
\quad Carrying capacity & """ + f"{pop.get('carrying_capacity', 0)*100:.1f}\\%" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Compute Scaling}} \\
"""
        
        if compute.get("by_particles"):
            first = compute["by_particles"][0]
            last = compute["by_particles"][-1]
            table_content += f"\\quad {first['n_particles']:,} particles & {first['wall_time_ms']:.0f} ms \\\\\n"
            table_content += f"\\quad {last['n_particles']:,} particles & {last['wall_time_ms']:.0f} ms \\\\\n"
        
        # Latency test (O(k) claim)
        if latency:
            table_content += r"\midrule" + "\n"
            table_content += r"\multicolumn{2}{l}{\textit{Latency vs Sequence Length (O(k) test)}} \\" + "\n"
            for lat in latency:
                table_content += f"\\quad N={lat['seq_len']:,} & {lat['ms_per_step']:.2f} ms/step \\\\\n"
        
        table_content += r"""\midrule
\multicolumn{2}{l}{\textit{Generalization}} \\
"""
        
        for g in gen:
            table_content += f"\\quad {g['name'].replace('_', ' ').title()} & {g['n_crystallized']} crystallized \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class ScalingDynamicsFigureProjector(BaseProjector):
    """Projector for scaling dynamics figure (4 panels)."""
    
    def __init__(
        self,
        config: ScalingFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = ScalingFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 4-panel dynamics visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        result = results[0]
        pop = result.get("population", {})
        history = pop.get("history", {})
        interference = result.get("interference", [])
        gen = result.get("generalization", [])
        
        self.ensure_output_dir()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if history:
            steps = history["step"]
            
            # Panel A: Mode population over time
            ax = axes[0, 0]
            ax.plot(steps, history["n_modes"], label="Total", color='#336699', linewidth=2)
            ax.plot(steps, history["n_crystallized"], label="Crystallized", color='#27ae60', linewidth=2)
            ax.plot(steps, history["n_stable"], label="Stable", color='#f39c12', linewidth=2, linestyle='--')
            ax.plot(steps, history["n_volatile"], label="Volatile", color='#e74c3c', linewidth=2, linestyle=':')
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel("Mode count", fontsize=10)
            ax.legend(loc='right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
            
            # Panel B: Birth/death rates
            ax = axes[0, 1]
            ax.bar(steps, history["n_births"], alpha=0.7, color='#27ae60', label="Births", width=1)
            ax.bar(steps, [-d for d in history["n_deaths"]], alpha=0.7, color='#e74c3c', label="Deaths", width=1)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel("Birth/Death count", fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Interference test
        ax = axes[1, 0]
        if interference:
            n_patterns = [r["n_patterns"] for r in interference]
            n_crystallized = [r["n_crystallized"] for r in interference]
            
            ax.plot(n_patterns, n_crystallized, 'o-', color='#336699', linewidth=2, markersize=8, label="Crystallized")
            ax.plot(n_patterns, n_patterns, '--', color='gray', linewidth=1, alpha=0.5, label="Ideal (1:1)")
            ax.set_xlabel("Number of distinct patterns", fontsize=10)
            ax.set_ylabel("Crystallized carriers", fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Panel D: Generalization
        ax = axes[1, 1]
        if gen:
            names = [g["name"].replace("_", "\n") for g in gen]
            structure = [g["structure_ratio"] for g in gen]
            colors = ['#27ae60' if s > 0.3 else '#f39c12' if s > 0.1 else '#e74c3c' for s in structure]
            
            ax.bar(range(len(names)), structure, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel("Structure ratio (crystallized / max)", fontsize=10)
            ax.set_ylim(0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"scaling_dynamics.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}


class ScalingComputeFigureProjector(BaseProjector):
    """Projector for scaling compute figure (3 panels)."""
    
    def __init__(
        self,
        config: ScalingFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = ScalingFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 3-panel compute scaling visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        result = results[0]
        compute = result.get("compute", {})
        latency = result.get("latency", [])
        
        self.ensure_output_dir()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel A: Time vs particle count
        ax = axes[0]
        by_particles = compute.get("by_particles", [])
        if by_particles:
            particles = [r["n_particles"] for r in by_particles]
            times = [r["wall_time_ms"] for r in by_particles]
            
            ax.plot(particles, times, 'o-', color='#336699', linewidth=2, markersize=8)
            ax.set_xlabel("Particle count", fontsize=10)
            ax.set_ylabel("Wall-clock time (ms)", fontsize=10)
            
            # Fit line to check scaling
            if len(particles) > 2:
                z = np.polyfit(particles, times, 1)
                p = np.poly1d(z)
                ax.plot(particles, p(particles), '--', color='gray', alpha=0.5, 
                       label=f"Linear fit: {z[0]:.4f}ms/particle")
                ax.legend(loc='upper left', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: Time vs grid size
        ax = axes[1]
        by_grid = compute.get("by_grid", [])
        if by_grid:
            cells = [r["grid_cells"] for r in by_grid]
            times = [r["wall_time_ms"] for r in by_grid]
            labels = [f"{r['grid_size'][0]}³" for r in by_grid]
            
            ax.bar(range(len(cells)), times, color='#4C994C', edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_xlabel("Grid size", fontsize=10)
            ax.set_ylabel("Wall-clock time (ms)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Latency vs sequence length (O(k) test)
        ax = axes[2]
        if latency:
            seq_lens = [r["seq_len"] for r in latency]
            ms_per_step = [r["ms_per_step"] for r in latency]
            
            ax.plot(seq_lens, ms_per_step, 'o-', color='#9b59b6', linewidth=2, markersize=8)
            ax.set_xlabel("Sequence length N", fontsize=10)
            ax.set_ylabel("Latency (ms/step)", fontsize=10)
            
            # Show mean line
            mean_latency = np.mean(ms_per_step)
            ax.axhline(y=mean_latency, color='gray', linestyle='--', alpha=0.5,
                      label=f"Mean: {mean_latency:.2f} ms/step")
            ax.legend(loc='upper left', fontsize=9)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"scaling_compute.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
