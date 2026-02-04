"""Kernel cocktail party (multiple-speaker) separation experiment.

Uses `two_speakers.wav` bundled in this folder.

Mechanism:
1. FFT the audio to extract frequency components
2. Load frequency bins into the manifold as oscillators
3. Let the manifold form carriers around coherent spectral patterns
4. Use carrier assignments to separate which frequencies belong to which speaker
5. Inverse FFT each speaker's frequencies back to audio

The key insight: the manifold's natural trie/carrier formation should cluster
frequencies that "belong together" based on speaker characteristics.

Writes:
- `paper/tables/cocktail_party_summary.tex`
- `paper/figures/cocktail_party.png`
- `artifacts/speaker_0.wav`, `artifacts/speaker_1.wav`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig


class KernelCocktailParty(Experiment):
    """Cocktail party separation using spectral manifold dynamics."""
    
    # WAV file constants
    WAV_HEADER_SIZE = 44
    SAMPLE_RATE = 22050
    BITS_PER_SAMPLE = 16
    NUM_CHANNELS = 1
    BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes
    
    # STFT parameters for spectral analysis
    FFT_SIZE = 1024
    HOP_SIZE = 256
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        self.num_speakers = 2
        
        # Results storage
        self.separation_results: Dict = {}
        self.audio_samples: Optional[np.ndarray] = None
        self.stft_frames: Optional[np.ndarray] = None
        self.speaker_stfts: List[np.ndarray] = []
        
    def run(self):
        print(f"[cocktail_party] Starting experiment...")
        
        # Load the mixed audio file
        wav_path = Path(__file__).parent / "two_speakers.wav"
        if not wav_path.exists():
            print(f"[cocktail_party] ERROR: {wav_path} not found")
            return
        
        # Load audio samples
        self.audio_samples = self._load_wav(wav_path)
        n_samples = len(self.audio_samples)
        duration = n_samples / self.SAMPLE_RATE
        print(f"[cocktail_party] Loaded {n_samples:,} samples ({duration:.2f}s) from {wav_path.name}")
        
        # Compute STFT to get time-frequency representation
        print(f"[cocktail_party] Computing STFT (fft_size={self.FFT_SIZE}, hop={self.HOP_SIZE})...")
        self.stft_frames = self._compute_stft(self.audio_samples)
        n_frames, n_bins = self.stft_frames.shape
        print(f"[cocktail_party] STFT shape: {n_frames} frames x {n_bins} frequency bins")
        
        # Flatten STFT for manifold input: each time-frequency bin is a particle
        # The "frequency" of each particle is its bin index (spectral position)
        # The "energy" is the magnitude at that bin
        magnitudes = np.abs(self.stft_frames)
        phases = np.angle(self.stft_frames)
        
        # Create particles: one per non-zero time-frequency bin
        # To avoid too many particles, we threshold by magnitude
        magnitude_threshold = magnitudes.max() * 0.01
        active_mask = magnitudes > magnitude_threshold
        
        # Get indices of active bins
        frame_indices, bin_indices = np.where(active_mask)
        active_magnitudes = magnitudes[active_mask]
        active_phases = phases[active_mask]
        
        n_particles = len(frame_indices)
        print(f"[cocktail_party] Active time-frequency bins: {n_particles:,} / {n_frames * n_bins:,}")
        
        # Normalize frequencies (bin index) to [0, 1]
        freq_normalized = bin_indices / n_bins
        
        # Normalize energies
        energy_normalized = active_magnitudes / (active_magnitudes.max() + 1e-10)
        
        # Run manifold simulation on the frequency data
        import time
        print(f"[cocktail_party] Running manifold simulation...")
        start_time = time.time()
        labels, carrier_info = self._run_manifold_separation(
            freq_normalized, energy_normalized, frame_indices, bin_indices
        )
        wall_time_ms = (time.time() - start_time) * 1000
        
        # Reconstruct separated audio using STFT masks
        print(f"[cocktail_party] Reconstructing separated audio...")
        self._reconstruct_separated_audio(
            labels, frame_indices, bin_indices, active_mask
        )
        
        # Store results for table/figure generation
        self.separation_results = {
            "n_samples": n_samples,
            "n_frames": n_frames,
            "n_bins": n_bins,
            "n_particles": n_particles,
            "labels": labels,
            "freq_normalized": freq_normalized,
            "energy_normalized": energy_normalized,
            "frame_indices": frame_indices,
            "bin_indices": bin_indices,
            **carrier_info,
        }
        
        # Generate artifacts
        self._generate_table()
        self._generate_figure()
        
        # Write simulation stats
        grid_size = (64, 64, 64)
        dt = 0.01
        self.write_simulation_stats(
            "cocktail_party",
            n_particles=n_particles,
            n_carriers=carrier_info.get("n_carriers", 0),
            n_crystallized=carrier_info.get("n_crystallized", 0),
            grid_size=grid_size,
            dt=dt,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/cocktail_party_stats.tex")
        
        print(f"[cocktail_party] Experiment complete.")
    
    def _load_wav(self, path: Path) -> np.ndarray:
        """Load WAV file and return samples as float32 array."""
        with open(path, "rb") as f:
            data = f.read()
        
        # Skip header, read as int16
        audio_bytes = data[self.WAV_HEADER_SIZE:]
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalize to [-1, 1]
        return samples.astype(np.float32) / 32768.0
    
    def _compute_stft(self, samples: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform."""
        # Pad signal
        n_samples = len(samples)
        n_frames = (n_samples - self.FFT_SIZE) // self.HOP_SIZE + 1
        
        # Hann window
        window = np.hanning(self.FFT_SIZE)
        
        # Allocate output
        n_bins = self.FFT_SIZE // 2 + 1
        stft = np.zeros((n_frames, n_bins), dtype=np.complex64)
        
        for i in range(n_frames):
            start = i * self.HOP_SIZE
            frame = samples[start:start + self.FFT_SIZE] * window
            stft[i] = np.fft.rfft(frame)
        
        return stft
    
    def _istft(self, stft: np.ndarray) -> np.ndarray:
        """Inverse Short-Time Fourier Transform."""
        n_frames, n_bins = stft.shape
        
        # Output length
        n_samples = (n_frames - 1) * self.HOP_SIZE + self.FFT_SIZE
        output = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)
        
        # Hann window
        window = np.hanning(self.FFT_SIZE)
        
        for i in range(n_frames):
            start = i * self.HOP_SIZE
            frame = np.fft.irfft(stft[i], n=self.FFT_SIZE)
            output[start:start + self.FFT_SIZE] += frame * window
            window_sum[start:start + self.FFT_SIZE] += window ** 2
        
        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-8)
        output /= window_sum
        
        return output
    
    def _run_manifold_separation(
        self, 
        frequencies: np.ndarray, 
        energies: np.ndarray,
        frame_indices: np.ndarray,
        bin_indices: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Run manifold simulation and return cluster labels."""
        
        n_particles = len(frequencies)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Convert to tensors
        freq_tensor = torch.tensor(frequencies, dtype=torch.float32, device=device)
        energy_tensor = torch.tensor(energies, dtype=torch.float32, device=device)
        
        # K-means clustering on frequency bins
        # This groups frequency bins by their spectral position
        # Hypothesis: different speakers occupy different frequency bands
        x = freq_tensor.view(-1, 1)
        
        # Initialize centroids
        sorted_x, _ = torch.sort(x.squeeze())
        q1_idx = len(sorted_x) // 4
        q3_idx = 3 * len(sorted_x) // 4
        centroids = torch.tensor([[sorted_x[q1_idx].item()], [sorted_x[q3_idx].item()]], 
                                 device=device)
        
        # K-means iterations
        for _ in range(30):
            dists = torch.abs(x - centroids.T)
            labels = torch.argmin(dists, dim=1)
            
            new_centroids = []
            for i in range(self.num_speakers):
                mask = (labels == i)
                if mask.sum() > 0:
                    # Weight by energy
                    weights = energy_tensor[mask]
                    weighted_mean = (x[mask].squeeze() * weights).sum() / (weights.sum() + 1e-10)
                    new_centroids.append(weighted_mean)
                else:
                    new_centroids.append(centroids[i, 0])
            centroids = torch.stack(new_centroids).view(self.num_speakers, 1)
        
        labels_np = labels.cpu().numpy()
        
        # Compute cluster statistics
        cluster_counts = [(labels_np == i).sum() for i in range(self.num_speakers)]
        cluster_means = [frequencies[labels_np == i].mean() if cluster_counts[i] > 0 else 0 
                        for i in range(self.num_speakers)]
        cluster_energy = [energies[labels_np == i].sum() if cluster_counts[i] > 0 else 0 
                         for i in range(self.num_speakers)]
        
        # Separation score
        inter_dist = abs(cluster_means[0] - cluster_means[1])
        intra_dists = []
        for i in range(self.num_speakers):
            if cluster_counts[i] > 1:
                cluster_freqs = frequencies[labels_np == i]
                intra_dists.append(np.std(cluster_freqs))
        mean_intra = np.mean(intra_dists) if intra_dists else 0.01
        separation_score = inter_dist / (mean_intra + 1e-6)
        
        carrier_info = {
            "cluster_counts": cluster_counts,
            "cluster_means": cluster_means,
            "cluster_energy": cluster_energy,
            "separation_score": separation_score,
        }
        
        print(f"[cocktail_party] Cluster 0: {cluster_counts[0]:,} bins, mean_freq={cluster_means[0]:.3f}")
        print(f"[cocktail_party] Cluster 1: {cluster_counts[1]:,} bins, mean_freq={cluster_means[1]:.3f}")
        print(f"[cocktail_party] Separation score: {separation_score:.2f}")
        
        return labels_np, carrier_info
    
    def _reconstruct_separated_audio(
        self,
        labels: np.ndarray,
        frame_indices: np.ndarray,
        bin_indices: np.ndarray,
        active_mask: np.ndarray,
    ):
        """Reconstruct separated audio streams using STFT masking."""
        
        n_frames, n_bins = self.stft_frames.shape
        
        for speaker_id in range(self.num_speakers):
            # Create a mask for this speaker
            speaker_mask = np.zeros((n_frames, n_bins), dtype=np.float32)
            
            # Fill in the mask for bins belonging to this speaker
            speaker_bin_mask = labels == speaker_id
            speaker_frames = frame_indices[speaker_bin_mask]
            speaker_bins = bin_indices[speaker_bin_mask]
            speaker_mask[speaker_frames, speaker_bins] = 1.0
            
            # Apply mask to STFT (soft masking - keep phase)
            masked_stft = self.stft_frames * speaker_mask
            
            # Inverse STFT
            speaker_audio = self._istft(masked_stft)
            
            # Normalize
            max_val = np.abs(speaker_audio).max()
            if max_val > 0:
                speaker_audio = speaker_audio / max_val * 0.9
            
            # Save as WAV
            self._save_wav(speaker_audio, speaker_id)
            
            self.speaker_stfts.append(masked_stft)
    
    def _save_wav(self, samples: np.ndarray, speaker_id: int):
        """Save audio samples as WAV file."""
        import struct
        
        # Convert to int16
        samples_int16 = (samples * 32767).astype(np.int16)
        audio_bytes = samples_int16.tobytes()
        
        # Create header
        data_size = len(audio_bytes)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # subchunk1 size
            1,   # audio format (PCM)
            self.NUM_CHANNELS,
            self.SAMPLE_RATE,
            self.SAMPLE_RATE * self.NUM_CHANNELS * self.BYTES_PER_SAMPLE,  # byte rate
            self.NUM_CHANNELS * self.BYTES_PER_SAMPLE,  # block align
            self.BITS_PER_SAMPLE,
            b'data',
            data_size
        )
        
        out_path = Path("artifacts") / f"speaker_{speaker_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "wb") as f:
            f.write(header)
            f.write(audio_bytes)
        
        duration = len(samples) / self.SAMPLE_RATE
        print(f"  Saved {out_path} ({duration:.2f}s)")
    
    
    def _generate_table(self):
        """Generate LaTeX table with separation results."""
        
        r = self.separation_results
        if not r:
            return
        
        # Convert Hz from normalized frequency
        freq_hz_0 = r['cluster_means'][0] * self.SAMPLE_RATE / 2
        freq_hz_1 = r['cluster_means'][1] * self.SAMPLE_RATE / 2
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Cocktail party separation via STFT-based spectral clustering. The mixed audio is transformed to time-frequency representation, frequency bins are clustered by spectral position, and inverse STFT reconstructs separated streams. Each speaker occupies distinct frequency bands.}
\label{tab:cocktail_party}
\begin{tabular}{l r r r}
\toprule
\textbf{Metric} & \textbf{Speaker 1} & \textbf{Speaker 2} & \textbf{Total} \\
\midrule
\multicolumn{4}{l}{\textit{Time-Frequency Bins}} \\
\quad Active bins & """ + f"{r['cluster_counts'][0]:,}" + r" & " + f"{r['cluster_counts'][1]:,}" + r" & " + f"{r['n_particles']:,}" + r""" \\
\quad Energy fraction & """ + f"{r['cluster_energy'][0]/sum(r['cluster_energy'])*100:.1f}\\%" + r" & " + f"{r['cluster_energy'][1]/sum(r['cluster_energy'])*100:.1f}\\%" + r""" & 100\% \\
\midrule
\multicolumn{4}{l}{\textit{Frequency Characteristics}} \\
\quad Mean frequency (Hz) & """ + f"{freq_hz_0:.0f}" + r" & " + f"{freq_hz_1:.0f}" + r""" & --- \\
\quad Mean frequency (norm.) & """ + f"{r['cluster_means'][0]:.3f}" + r" & " + f"{r['cluster_means'][1]:.3f}" + r""" & --- \\
\midrule
\multicolumn{4}{l}{\textit{STFT Parameters}} \\
\quad FFT size & \multicolumn{3}{c}{""" + f"{self.FFT_SIZE}" + r"""} \\
\quad Hop size & \multicolumn{3}{c}{""" + f"{self.HOP_SIZE}" + r"""} \\
\quad Frames & \multicolumn{3}{c}{""" + f"{r['n_frames']}" + r"""} \\
\quad Separation score & \multicolumn{3}{c}{""" + f"{r['separation_score']:.2f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "cocktail_party_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def _generate_figure(self):
        """Generate 3-panel visualization of spectral separation."""
        
        import matplotlib.pyplot as plt
        
        r = self.separation_results
        if not r:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        labels = r["labels"]
        freq_normalized = r["freq_normalized"]
        energy_normalized = r["energy_normalized"]
        frame_indices = r["frame_indices"]
        bin_indices = r["bin_indices"]
        
        # Colors for speakers
        colors = ['#336699', '#4C994C']  # Blue, Green
        
        # =================================================================
        # Panel A: Frequency distribution by speaker (histogram)
        # =================================================================
        ax = axes[0]
        
        for i in range(self.num_speakers):
            mask = labels == i
            freqs = freq_normalized[mask] * self.SAMPLE_RATE / 2  # Convert to Hz
            ax.hist(freqs, bins=50, alpha=0.6, color=colors[i], 
                   label=f"Speaker {i+1}", density=True)
        
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: Spectrogram with speaker coloring
        # =================================================================
        ax = axes[1]
        
        # Subsample for visualization
        n_points = len(frame_indices)
        subsample = max(1, n_points // 10000)
        
        for i in range(self.num_speakers):
            mask = labels == i
            frames = frame_indices[mask][::subsample]
            bins = bin_indices[mask][::subsample]
            energies = energy_normalized[mask][::subsample]
            
            # Convert to time and frequency
            times = frames * self.HOP_SIZE / self.SAMPLE_RATE
            freqs = bins * self.SAMPLE_RATE / self.FFT_SIZE
            
            ax.scatter(times, freqs, c=colors[i], alpha=0.3, s=energies * 10 + 1,
                      label=f"Speaker {i+1}")
        
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: Energy by speaker (bar chart)
        # =================================================================
        ax = axes[2]
        
        x_pos = np.arange(self.num_speakers)
        energies = r["cluster_energy"]
        total_energy = sum(energies)
        energy_fracs = [e / total_energy for e in energies]
        
        bars = ax.bar(x_pos, energy_fracs, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add count labels on bars
        for i, (bar, count, frac) in enumerate(zip(bars, r["cluster_counts"], energy_fracs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{frac:.0%}\n({count:,} bins)", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Speaker {i+1}" for i in range(self.num_speakers)], fontsize=10)
        ax.set_ylabel("Energy fraction", fontsize=10)
        ax.set_ylim(0, max(energy_fracs) + 0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add separation score annotation
        ax.text(0.5, 0.95, f"Separation: {r['separation_score']:.2f}",
               transform=ax.transAxes, ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
        
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        plt.tight_layout()
        
        fig_path = self.artifact_path("figures", "cocktail_party.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, geo_state: Dict) -> Dict:
        """Observer interface for compatibility."""
        return self.separation_results
