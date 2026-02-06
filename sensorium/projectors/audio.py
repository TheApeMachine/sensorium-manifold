"""Audio experiment projectors.

Custom projectors for audio waveform visualization and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING

import numpy as np

from sensorium.projectors.base import BaseProjector
from sensorium.dataset.audio import dequantize_bytes_to_audio

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class AudioFigureConfig:
    """Configuration for audio figure.
    
    Attributes:
        name: Output filename (without extension)
        sample_rate: Audio sample rate for time axis
        format: Output format (png, pdf, svg)
        dpi: Output DPI
    """
    name: str = "audio_gen"
    sample_rate: int = 8000
    format: str = "png"
    dpi: int = 300


class AudioTableProjector(BaseProjector):
    """Projector for audio generation summary table.
    
    Generates a LaTeX table showing MAE, SNR, and accuracy.
    
    Expects InferenceObserver results with fields:
    - waveform_type: Type of waveform
    - mask_frac: Mask fraction
    - mae: Mean absolute error
    - snr: Signal-to-noise ratio
    - accuracy: Byte-level accuracy
    """
    
    def __init__(self, output_dir: Path | None = None, name: str = "audio_gen_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX summary table."""
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        self.ensure_output_dir()
        
        # Group by waveform type
        by_waveform: Dict[str, List[Dict]] = {}
        for r in results:
            wf = r.get("waveform_type", "unknown")
            if wf not in by_waveform:
                by_waveform[wf] = []
            by_waveform[wf].append(r)
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Audio waveform inpainting results. Byte-quantized samples are reconstructed using periodic position matching. SNR measures signal quality; MAE measures amplitude error.}
\label{tab:audio_gen}
\begin{tabular}{l l c c c}
\toprule
\textbf{Waveform} & \textbf{Mask} & \textbf{MAE} & \textbf{SNR (dB)} & \textbf{Accuracy} \\
\midrule
"""
        for wf_type, wf_results in by_waveform.items():
            wf_results = sorted(wf_results, key=lambda r: r.get("mask_frac", 0))
            for i, res in enumerate(wf_results):
                wf_name = wf_type.title() if i == 0 else ""
                mask_frac = res.get("mask_frac", 0)
                mae = res.get("mae", 0)
                snr = res.get("snr", 0)
                accuracy = res.get("accuracy", 0)
                
                table_content += f"{wf_name} & {mask_frac*100:.0f}\\% & "
                table_content += f"{mae:.4f} & "
                table_content += f"{snr:.1f} & "
                table_content += f"{accuracy*100:.1f}\\% \\\\\n"
            
            # Add spacing between waveform types
            if wf_type != list(by_waveform.keys())[-1]:
                table_content += r"\addlinespace" + "\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(table_content)
        
        return {"status": "success", "path": str(output_path)}


class AudioFigureProjector(BaseProjector):
    """Projector for audio 3-panel visualization.
    
    Generates:
    A) Waveform comparison (original vs reconstructed)
    B) SNR by waveform type
    C) MAE vs mask level
    
    Expects InferenceObserver results with fields:
    - waveform_type, mask_frac, mae, snr, accuracy
    - original_bytes, reconstructed_bytes (optional for panel A)
    """
    
    def __init__(
        self,
        config: AudioFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        
        if config:
            self.config = config
        else:
            self.config = AudioFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 3-panel visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "skipped", "reason": "no results"}
        
        self.ensure_output_dir()
        
        # Group by waveform type
        by_waveform: Dict[str, List[Dict]] = {}
        for r in results:
            wf = r.get("waveform_type", "unknown")
            if wf not in by_waveform:
                by_waveform[wf] = []
            by_waveform[wf].append(r)
        
        waveform_types = list(by_waveform.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel A: Waveform comparison
        ax = axes[0]
        # Find an example with original/reconstructed bytes
        example = None
        for r in results:
            if r.get("original_bytes") and r.get("reconstructed_bytes"):
                example = r
                break
        
        if example:
            original = dequantize_bytes_to_audio(example["original_bytes"])[:200]
            recon = dequantize_bytes_to_audio(example["reconstructed_bytes"])[:200]
            
            t = np.arange(len(original)) / self.config.sample_rate * 1000
            ax.plot(t, original, label='Original', alpha=0.8, linewidth=1.5, color='#336699')
            ax.plot(t, recon, label='Reconstructed', alpha=0.8, linewidth=1.5, 
                   linestyle='--', color='#4C994C')
            ax.set_xlabel("Time (ms)", fontsize=10)
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: SNR by waveform type (at 20% mask)
        ax = axes[1]
        wf_names = [wf.title() for wf in waveform_types]
        snrs = []
        for wf in waveform_types:
            wf_results = by_waveform.get(wf, [])
            # Find result with mask_frac closest to 0.2
            snr_val = 0
            for r in wf_results:
                if abs(r.get("mask_frac", 0) - 0.2) < 0.05:
                    snr_val = r.get("snr", 0)
                    break
            snrs.append(snr_val)
        
        colors = ['#336699', '#4C994C', '#CC6633', '#9966CC']
        bars = ax.bar(range(len(wf_names)), snrs, color=colors[:len(wf_names)],
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        for bar, val in zip(bars, snrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:.1f}", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(wf_names)))
        ax.set_xticklabels(wf_names, fontsize=9)
        ax.set_ylabel("SNR (dB)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: MAE vs mask level for first waveform type
        ax = axes[2]
        if waveform_types:
            first_wf = waveform_types[0]
            wf_results = sorted(by_waveform.get(first_wf, []), key=lambda r: r.get("mask_frac", 0))
            mask_pcts = [r.get("mask_frac", 0) * 100 for r in wf_results]
            maes = [r.get("mae", 0) for r in wf_results]
            
            ax.plot(mask_pcts, maes, 'o-', color='#336699', linewidth=2, markersize=10)
        
        ax.set_xlabel("Mask percentage", fontsize=10)
        ax.set_ylabel("MAE", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
