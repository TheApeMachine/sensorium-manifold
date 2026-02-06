"""Cocktail party projectors for tables and figures.

Generates summary tables and visualization figures for speaker separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class CocktailPartyFigureConfig:
    """Configuration for cocktail party figure."""
    name: str = "cocktail_party"
    format: str = "png"
    dpi: int = 300
    sample_rate: int = 22050
    fft_size: int = 1024
    hop_size: int = 256


class CocktailPartyTableProjector(BaseProjector):
    """Projector for cocktail party summary table."""
    
    def __init__(
        self,
        output_dir: Path | None = None,
        name: str = "cocktail_party_summary",
        sample_rate: int = 22050,
        fft_size: int = 1024,
        hop_size: int = 256,
    ):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        r = results[0]
        
        self.ensure_output_dir()
        
        cluster_means = r.get("cluster_means", [0, 0])
        cluster_counts = r.get("cluster_counts", [0, 0])
        cluster_energy = r.get("cluster_energy", [0, 0])
        n_particles = r.get("n_particles", 0)
        n_frames = r.get("n_frames", 0)
        separation_score = r.get("separation_score", 0)
        
        freq_hz_0 = cluster_means[0] * self.sample_rate / 2
        freq_hz_1 = cluster_means[1] * self.sample_rate / 2
        total_energy = sum(cluster_energy) or 1
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Cocktail party separation via STFT-based spectral clustering. The mixed audio is transformed to time-frequency representation, frequency bins are clustered by spectral position, and inverse STFT reconstructs separated streams. Each speaker occupies distinct frequency bands.}
\label{tab:cocktail_party}
\begin{tabular}{l r r r}
\toprule
\textbf{Metric} & \textbf{Speaker 1} & \textbf{Speaker 2} & \textbf{Total} \\
\midrule
\multicolumn{4}{l}{\textit{Time-Frequency Bins}} \\
\quad Active bins & """ + f"{cluster_counts[0]:,}" + r" & " + f"{cluster_counts[1]:,}" + r" & " + f"{n_particles:,}" + r""" \\
\quad Energy fraction & """ + f"{cluster_energy[0]/total_energy*100:.1f}\\%" + r" & " + f"{cluster_energy[1]/total_energy*100:.1f}\\%" + r""" & 100\% \\
\midrule
\multicolumn{4}{l}{\textit{Frequency Characteristics}} \\
\quad Mean frequency (Hz) & """ + f"{freq_hz_0:.0f}" + r" & " + f"{freq_hz_1:.0f}" + r""" & --- \\
\quad Mean frequency (norm.) & """ + f"{cluster_means[0]:.3f}" + r" & " + f"{cluster_means[1]:.3f}" + r""" & --- \\
\midrule
\multicolumn{4}{l}{\textit{STFT Parameters}} \\
\quad FFT size & \multicolumn{3}{c}{""" + f"{self.fft_size}" + r"""} \\
\quad Hop size & \multicolumn{3}{c}{""" + f"{self.hop_size}" + r"""} \\
\quad Frames & \multicolumn{3}{c}{""" + f"{n_frames}" + r"""} \\
\quad Separation score & \multicolumn{3}{c}{""" + f"{separation_score:.2f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class CocktailPartyFigureProjector(BaseProjector):
    """Projector for cocktail party 3-panel figure."""
    
    def __init__(
        self,
        config: CocktailPartyFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = CocktailPartyFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 3-panel visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        r = results[0]
        
        labels = r.get("labels")
        freq_normalized = r.get("freq_normalized")
        energy_normalized = r.get("energy_normalized")
        frame_indices = r.get("frame_indices")
        bin_indices = r.get("bin_indices")
        cluster_counts = r.get("cluster_counts", [0, 0])
        cluster_energy = r.get("cluster_energy", [0, 0])
        
        if labels is None or freq_normalized is None:
            return {"status": "skipped", "reason": "missing data"}
        
        self.ensure_output_dir()
        
        num_speakers = 2
        colors = ['#336699', '#4C994C']
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel A: Frequency distribution by speaker
        ax = axes[0]
        for i in range(num_speakers):
            mask = labels == i
            freqs = freq_normalized[mask] * self.config.sample_rate / 2
            ax.hist(freqs, bins=50, alpha=0.6, color=colors[i], 
                   label=f"Speaker {i+1}", density=True)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: Spectrogram with speaker coloring
        ax = axes[1]
        n_points = len(frame_indices)
        subsample = max(1, n_points // 10000)
        
        for i in range(num_speakers):
            mask = labels == i
            frames = frame_indices[mask][::subsample]
            bins = bin_indices[mask][::subsample]
            energies = energy_normalized[mask][::subsample]
            
            times = frames * self.config.hop_size / self.config.sample_rate
            freqs = bins * self.config.sample_rate / self.config.fft_size
            
            ax.scatter(times, freqs, c=colors[i], alpha=0.3, s=energies * 10 + 1,
                      label=f"Speaker {i+1}")
        
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Energy by speaker
        ax = axes[2]
        x_pos = np.arange(num_speakers)
        total_energy = sum(cluster_energy) or 1
        energy_fracs = [e / total_energy for e in cluster_energy]
        
        bars = ax.bar(x_pos, energy_fracs, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for i, (bar, count, frac) in enumerate(zip(bars, cluster_counts, energy_fracs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{frac:.0%}\n({count:,} bins)", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Speaker {i+1}" for i in range(num_speakers)], fontsize=10)
        ax.set_ylabel("Energy fraction", fontsize=10)
        ax.set_ylim(0, max(energy_fracs) + 0.15 if energy_fracs else 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
