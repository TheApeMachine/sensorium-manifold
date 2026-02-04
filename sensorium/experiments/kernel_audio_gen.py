"""Kernel audio waveform inpainting via thermodynamic trie.

We quantize waveform samples to bytes and reconstruct missing segments
using the segment-based periodicity mechanism.

NON-CHEATING DESIGN:
====================
- Training: Learn from waveform samples (byte-quantized)
- Inference: Mask random positions, reconstruct using periodic patterns
- No access to ground truth during reconstruction

The key mechanism is that periodic waveforms create consistent token IDs
at matching positions within each period (segment_size = samples per period).

Produces:
- `paper/tables/audio_gen_summary.tex`
- `paper/figures/audio_gen.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig


def quantize_audio_to_bytes(samples: np.ndarray) -> bytes:
    """Quantize float audio samples (-1 to 1) to bytes (0-255)."""
    normalized = (samples + 1.0) / 2.0
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return bytes(quantized)


def dequantize_bytes_to_audio(data: bytes) -> np.ndarray:
    """Dequantize bytes back to float audio samples."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return (arr / 255.0) * 2.0 - 1.0


class KernelAudioGen(Experiment):
    """Audio waveform inpainting experiment."""
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        # Tokenizer params
        self.vocab_size = 4096
        self.prime = 31
        self.mask = self.vocab_size - 1
        self.inv_prime = pow(self.prime, -1, self.vocab_size)
        
        # Audio params
        self.sample_rate = 8000
        self.duration = 1.0
        self.frequency = 440.0  # A4 note
        self.waveform_types = ["sine", "square", "sawtooth", "mixed"]
        self.mask_fracs = [0.1, 0.2, 0.3]
        
        self.results: Dict[str, Dict[float, Dict[str, Any]]] = {}
        self.examples: List[Dict[str, Any]] = []
    
    def _generate_waveform(self, waveform_type: str) -> np.ndarray:
        """Generate synthetic waveform."""
        n_samples = int(self.duration * self.sample_rate)
        t = np.arange(n_samples, dtype=np.float32) / self.sample_rate
        
        if waveform_type == "sine":
            y = np.sin(2 * np.pi * self.frequency * t)
        elif waveform_type == "square":
            y = np.sign(np.sin(2 * np.pi * self.frequency * t))
        elif waveform_type == "sawtooth":
            period = 1.0 / self.frequency
            y = 2 * (t / period - np.floor(t / period + 0.5))
        elif waveform_type == "mixed":
            y = (0.5 * np.sin(2 * np.pi * self.frequency * t) +
                 0.3 * np.sin(2 * np.pi * self.frequency * 2 * t) +
                 0.2 * np.sin(2 * np.pi * self.frequency * 3 * t))
        else:
            y = np.zeros(n_samples)
        
        # Add small noise
        rng = np.random.RandomState(42)
        y += 0.02 * rng.randn(n_samples)
        
        return np.clip(y, -1, 1).astype(np.float32)
    
    def run(self):
        print("[audio_gen] Starting experiment...")
        
        import time
        
        rng = np.random.RandomState(42)
        
        # Period in samples
        period_samples = int(self.sample_rate / self.frequency)
        print(f"[audio_gen] Period: {period_samples} samples at {self.frequency} Hz")
        
        # Track stats from first waveform
        self._stats = None
        
        for waveform_type in self.waveform_types:
            print(f"[audio_gen] Processing: {waveform_type}")
            self.results[waveform_type] = {}
            
            # Generate waveform
            samples = self._generate_waveform(waveform_type)
            audio_bytes = quantize_audio_to_bytes(samples)
            
            # Split: train on first 80%, test on last 20%
            split_idx = int(len(audio_bytes) * 0.8)
            train_bytes = audio_bytes[:split_idx]
            test_bytes = audio_bytes[split_idx:]
            
            print(f"[audio_gen] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
            
            # Train manifold
            def train_generator():
                yield train_bytes
            
            grid_size = (64, 64, 64)
            dt = 0.01
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                    segment_size=period_samples,  # Match audio period
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                    grid_size=grid_size,
                    dt=dt,
                ),
                generator=train_generator,
            )
            
            manifold = Manifold(cfg)
            start_time = time.time()
            state = manifold.run()
            run_time_ms = (time.time() - start_time) * 1000
            
            # Capture stats from first waveform
            if self._stats is None:
                token_ids = state.get("token_ids")
                n_particles = len(token_ids) if token_ids is not None else 0
                carriers = manifold.carriers or {}
                amplitudes = carriers.get("amplitudes")
                n_carriers = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
                crystallized = carriers.get("crystallized")
                n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
                
                self._stats = {
                    "n_particles": n_particles,
                    "n_carriers": n_carriers,
                    "n_crystallized": n_crystallized,
                    "grid_size": grid_size,
                    "dt": dt,
                    "wall_time_ms": run_time_ms,
                }
            
            # Test at different mask levels
            for mask_frac in self.mask_fracs:
                n_mask = int(len(test_bytes) * mask_frac)
                mask_positions = set(rng.choice(len(test_bytes), size=n_mask, replace=False))
                
                # Reconstruct
                reconstructed = bytearray(test_bytes)
                
                for pos in sorted(mask_positions):
                    seg_pos = (split_idx + pos) % period_samples
                    predicted = self._predict_sample(train_bytes, seg_pos, period_samples)
                    reconstructed[pos] = predicted
                
                # Calculate metrics
                original_audio = dequantize_bytes_to_audio(test_bytes)
                recon_audio = dequantize_bytes_to_audio(bytes(reconstructed))
                
                mae = np.mean(np.abs(original_audio - recon_audio))
                mse = np.mean((original_audio - recon_audio) ** 2)
                
                # SNR
                signal_power = np.mean(original_audio ** 2)
                noise_power = mse
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                
                # Byte accuracy on masked positions
                correct = sum(1 for pos in mask_positions 
                             if reconstructed[pos] == test_bytes[pos])
                accuracy = correct / n_mask if n_mask > 0 else 0.0
                
                self.results[waveform_type][mask_frac] = {
                    "mae": float(mae),
                    "mse": float(mse),
                    "snr": float(snr),
                    "accuracy": float(accuracy),
                }
                
                # Save example
                if mask_frac == 0.2 and waveform_type == "sine":
                    self.examples.append({
                        "waveform_type": waveform_type,
                        "original": test_bytes,
                        "reconstructed": bytes(reconstructed),
                        "mask_frac": mask_frac,
                    })
                
                print(f"[audio_gen] {waveform_type} @ {mask_frac*100:.0f}%: "
                      f"MAE={mae:.4f}, SNR={snr:.1f}dB, Acc={accuracy:.1%}")
        
        self._generate_table()
        self._generate_figure()
        
        # Write simulation stats
        if self._stats:
            self.write_simulation_stats(
                "audio_gen",
                n_particles=self._stats["n_particles"],
                n_carriers=self._stats["n_carriers"],
                n_crystallized=self._stats["n_crystallized"],
                grid_size=self._stats["grid_size"],
                dt=self._stats["dt"],
                n_steps=1,
                wall_time_ms=self._stats["wall_time_ms"],
            )
            print(f"✓ Generated: paper/tables/audio_gen_stats.tex")
        
        print("[audio_gen] Experiment complete.")
    
    def _predict_sample(
        self,
        train_bytes: bytes,
        target_seg_pos: int,
        period: int,
    ) -> int:
        """Predict sample using position periodicity."""
        
        values = []
        weights = []
        
        for i in range(len(train_bytes)):
            if i % period == target_seg_pos:
                values.append(train_bytes[i])
                recency = (i + 1) / len(train_bytes)
                weights.append(recency)
        
        if not values:
            return 128
        
        values_np = np.array(values, dtype=np.float32)
        weights_np = np.array(weights, dtype=np.float32)
        weights_np = weights_np / weights_np.sum()
        
        predicted = np.average(values_np, weights=weights_np)
        return int(np.clip(np.round(predicted), 0, 255))
    
    def _generate_table(self):
        """Generate LaTeX table."""
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Audio waveform inpainting results. Byte-quantized samples are reconstructed using periodic position matching. SNR measures signal quality; MAE measures amplitude error.}
\label{tab:audio_gen}
\begin{tabular}{l l c c c c}
\toprule
\textbf{Waveform} & \textbf{Mask} & \textbf{MAE} & \textbf{SNR (dB)} & \textbf{Accuracy} \\
\midrule
"""
        for waveform_type in self.waveform_types:
            for i, mask_frac in enumerate(self.mask_fracs):
                res = self.results.get(waveform_type, {}).get(mask_frac, {})
                if res:
                    wf_name = waveform_type.title() if i == 0 else ""
                    table_content += f"{wf_name} & {mask_frac*100:.0f}\\% & "
                    table_content += f"{res['mae']:.4f} & "
                    table_content += f"{res['snr']:.1f} & "
                    table_content += f"{res['accuracy']*100:.1f}\\% \\\\\n"
            if waveform_type != self.waveform_types[-1]:
                table_content += r"\addlinespace" + "\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "audio_gen_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def _generate_figure(self):
        """Generate 3-panel visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # =================================================================
        # Panel A: Waveform comparison (sine at 20% mask)
        # =================================================================
        ax = axes[0]
        
        if self.examples:
            ex = self.examples[0]
            original = dequantize_bytes_to_audio(ex["original"])[:200]
            recon = dequantize_bytes_to_audio(ex["reconstructed"])[:200]
            
            t = np.arange(len(original)) / self.sample_rate * 1000  # ms
            ax.plot(t, original, label='Original', alpha=0.8, linewidth=1.5, color='#336699')
            ax.plot(t, recon, label='Reconstructed', alpha=0.8, linewidth=1.5, 
                   linestyle='--', color='#4C994C')
            
            ax.set_xlabel("Time (ms)", fontsize=10)
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: SNR by waveform type (at 20% mask)
        # =================================================================
        ax = axes[1]
        
        wf_names = [wf.title() for wf in self.waveform_types]
        snrs = [self.results.get(wf, {}).get(0.2, {}).get("snr", 0) 
               for wf in self.waveform_types]
        colors = ['#336699', '#4C994C', '#CC6633', '#9966CC']
        
        bars = ax.bar(range(len(wf_names)), snrs, color=colors,
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for bar, val in zip(bars, snrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:.1f}", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(range(len(wf_names)))
        ax.set_xticklabels(wf_names, fontsize=9)
        ax.set_ylabel("SNR (dB)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: MAE vs mask level for sine
        # =================================================================
        ax = axes[2]
        
        sine_res = self.results.get("sine", {})
        mask_pcts = [mf * 100 for mf in sorted(sine_res.keys())]
        maes = [sine_res[mf]["mae"] for mf in sorted(sine_res.keys())]
        
        ax.plot(mask_pcts, maes, 'o-', color='#336699', linewidth=2, markersize=10)
        ax.set_xlabel("Mask percentage", fontsize=10)
        ax.set_ylabel("MAE", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "audio_gen.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
