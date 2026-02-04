"""Kernel audio handling via Universal Tokenizer (waveform inpainting demo).

We quantize waveform samples to bytes and reconstruct a missing segment by
sampling bytes, similar to the MNIST inpainting experiment.

NON-CHEATING DESIGN:
====================
This experiment uses a proper train/test split:
- Training: Learn from waveform samples (clean audio)
- Inference: Given partially masked audio, reconstruct missing samples
- No access to ground truth during reconstruction

The key mechanism is that periodic waveforms have repeating patterns that
create consistent token IDs at similar positions in the period.

Writes:
- `paper/tables/audio_gen_summary.tex`
- `paper/figures/audio_gen.pdf`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator

import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    SpectralSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.carrier import CarrierObserver


def quantize_audio_to_bytes(samples: np.ndarray) -> bytes:
    """Quantize float audio samples (-1 to 1) to bytes (0-255)."""
    # Normalize to [0, 255]
    normalized = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return bytes(quantized)


def dequantize_bytes_to_audio(data: bytes) -> np.ndarray:
    """Dequantize bytes back to float audio samples."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return (arr / 255.0) * 2.0 - 1.0  # [0, 255] -> [-1, 1]


class SyntheticAudio:
    """Generate synthetic audio waveforms with learnable patterns."""
    
    def __init__(
        self,
        duration_seconds: float = 2.0,
        sample_rate: int = 8000,  # Low sample rate for demo
        waveform_type: str = "sine",  # sine, square, sawtooth, mixed
        frequency: float = 440.0,  # A4 note
        seed: int = 42,
    ):
        self.duration = duration_seconds
        self.sample_rate = sample_rate
        self.waveform_type = waveform_type
        self.frequency = frequency
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        self.n_samples = int(duration_seconds * sample_rate)
        self.samples = self._generate()
        self.bytes_data = quantize_audio_to_bytes(self.samples)
    
    def _generate(self) -> np.ndarray:
        """Generate the audio waveform."""
        t = np.arange(self.n_samples, dtype=np.float32) / self.sample_rate
        
        if self.waveform_type == "sine":
            # Pure sine wave
            y = np.sin(2 * np.pi * self.frequency * t)
        
        elif self.waveform_type == "square":
            # Square wave
            y = np.sign(np.sin(2 * np.pi * self.frequency * t))
        
        elif self.waveform_type == "sawtooth":
            # Sawtooth wave
            period = 1.0 / self.frequency
            y = 2 * (t / period - np.floor(t / period + 0.5))
        
        elif self.waveform_type == "mixed":
            # Multiple harmonics (more complex pattern)
            y = (
                0.5 * np.sin(2 * np.pi * self.frequency * t) +
                0.3 * np.sin(2 * np.pi * self.frequency * 2 * t) +
                0.2 * np.sin(2 * np.pi * self.frequency * 3 * t)
            )
        
        elif self.waveform_type == "speech_like":
            # Pseudo-speech: amplitude modulated with varying frequency
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5 Hz modulation
            carrier = np.sin(2 * np.pi * self.frequency * (1 + 0.2 * np.sin(2 * np.pi * 3 * t)) * t)
            y = envelope * carrier
        
        else:
            raise ValueError(f"Unknown waveform type: {self.waveform_type}")
        
        # Add small noise
        y += 0.02 * self._rng.randn(self.n_samples)
        
        # Normalize to [-1, 1]
        y = np.clip(y, -1, 1)
        
        return y.astype(np.float32)
    
    def train_test_split(self, test_ratio: float = 0.2) -> Tuple[bytes, bytes]:
        """Split audio - training is BEFORE testing (no lookahead)."""
        split_idx = int(self.n_samples * (1 - test_ratio))
        
        train_bytes = self.bytes_data[:split_idx]
        test_bytes = self.bytes_data[split_idx:]
        
        return train_bytes, test_bytes


class AudioInpainter:
    """Reconstruct masked audio samples using dual-domain inference.
    
    For audio:
    - Spectral domain captures periodicity (carriers form at signal frequencies)
    - Temporal continuity provides smoothness prior
    """
    
    def __init__(self, vocab_size: int = 4096, prime: int = 31, period_samples: int = 0):
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask = vocab_size - 1
        self.period_samples = period_samples
        self.inference = None
    
    def learn_from_manifold(self, geo_state: Dict, spec_state: Dict):
        """Set up dual-domain inference from manifold state."""
        from sensorium.observers.dual_domain import DualDomainInference
        
        self.inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=spec_state,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
    
    def inpaint(
        self,
        corrupted: bytes,
        mask_positions: List[int],
        context_window: int = 10,
    ) -> bytes:
        """Reconstruct masked samples using dual-domain inference.
        
        Strategy:
        1. Use nearby unmasked samples as context (temporal locality)
        2. Find carriers they couple to (periodic patterns)
        3. Score candidates by carrier coupling + temporal smoothness
        """
        result = bytearray(corrupted)
        
        if self.inference is None:
            return bytes(result)
        
        for pos in mask_positions:
            # Get temporal context (nearby unmasked samples)
            context_positions = []
            for i in range(max(0, pos - context_window), min(len(result), pos + context_window + 1)):
                if i not in mask_positions and i != pos:
                    context_positions.append(i)
            
            if context_positions:
                context_indices = torch.tensor(
                    context_positions,
                    device=self.inference.device,
                    dtype=torch.int64
                )
                
                # Get carrier-based scores
                carrier_scores = self.inference.score_candidate_bytes(
                    context_indices=context_indices,
                    target_position=pos,
                    segment_size=self.period_samples if self.period_samples > 0 else None,
                )
            else:
                carrier_scores = np.ones(256, dtype=np.float32) / 256
            
            # Add temporal smoothness prior
            smoothness_scores = np.zeros(256, dtype=np.float32)
            
            # Look at immediate neighbors
            if pos > 0 and pos - 1 not in mask_positions:
                prev_val = result[pos - 1]
                for sval in range(256):
                    diff = abs(sval - prev_val)
                    smoothness_scores[sval] += np.exp(-diff**2 / (2 * 20**2))
            
            if pos < len(result) - 1 and pos + 1 not in mask_positions:
                next_val = result[pos + 1]
                for sval in range(256):
                    diff = abs(sval - next_val)
                    smoothness_scores[sval] += np.exp(-diff**2 / (2 * 20**2))
            
            if smoothness_scores.sum() > 0:
                smoothness_scores /= smoothness_scores.sum()
            
            # Combined: carrier coupling (spectral) + smoothness (temporal)
            combined = 0.5 * carrier_scores + 0.5 * smoothness_scores
            
            result[pos] = int(np.argmax(combined))
        
        return bytes(result)


class KernelAudioGen(Experiment):
    """Audio waveform inpainting experiment using Universal Tokenizer."""
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        
        self.sample_rate = 8000
        self.duration = 1.0  # seconds
        self.waveform_types = ["sine", "square", "sawtooth", "mixed"]
        self.mask_fracs = [0.1, 0.2, 0.3]
        
        self.results: Dict[str, Dict[float, Dict[str, Any]]] = {}
        self.examples: List[Dict[str, Any]] = []

    def observe(self, state: dict):
        """Generate paper artifacts."""
        if not self.results:
            print("Warning: No results collected")
            return
        
        import matplotlib.pyplot as plt
        
        # Summary table
        summary = {}
        for waveform_type, frac_results in self.results.items():
            for mask_frac, res in frac_results.items():
                key = f"{waveform_type}_{int(mask_frac*100)}pct"
                summary[f"{key}_snr"] = res.get("snr", 0.0)
                summary[f"{key}_mae"] = res.get("mae", 0.0)
        
        self.write_kv_table("audio_gen_summary", summary)
        
        # Figure: Waveform comparison for each type
        n_types = len(self.waveform_types)
        fig, axes = plt.subplots(n_types, 2, figsize=(14, 3 * n_types))
        if n_types == 1:
            axes = axes.reshape(1, -1)
        
        for idx, waveform_type in enumerate(self.waveform_types):
            # Find an example for this type
            example = next(
                (e for e in self.examples if e.get("waveform_type") == waveform_type),
                None
            )
            
            if example is None:
                continue
            
            # Left: Waveform comparison
            ax = axes[idx, 0]
            original = dequantize_bytes_to_audio(example["original"])
            reconstructed = dequantize_bytes_to_audio(example["reconstructed"])
            
            t = np.arange(len(original)) / self.sample_rate * 1000  # ms
            
            ax.plot(t[:1000], original[:1000], label='Original', alpha=0.8, linewidth=1)
            ax.plot(t[:1000], reconstructed[:1000], label='Reconstructed', 
                   alpha=0.8, linewidth=1, linestyle='--')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{waveform_type.title()} Waveform Reconstruction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Right: Error over mask fractions
            ax = axes[idx, 1]
            frac_results = self.results.get(waveform_type, {})
            fracs = sorted(frac_results.keys())
            maes = [frac_results[f]["mae"] for f in fracs]
            snrs = [frac_results[f]["snr"] for f in fracs]
            
            ax2 = ax.twinx()
            line1 = ax.plot([f*100 for f in fracs], maes, 'o-', color='#336699',
                           linewidth=2, markersize=8, label='MAE')
            line2 = ax2.plot([f*100 for f in fracs], snrs, 's-', color='#4C994C',
                            linewidth=2, markersize=8, label='SNR (dB)')
            
            ax.set_xlabel('Mask Percentage (%)')
            ax.set_ylabel('MAE', color='#336699')
            ax2.set_ylabel('SNR (dB)', color='#4C994C')
            ax.set_title(f'{waveform_type.title()}: Metrics vs Corruption')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        
        plt.suptitle('Audio Waveform Inpainting via Universal Tokenizer',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "audio_gen.pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/audio_gen_summary.tex")

    def run(self):
        """Run audio inpainting experiment."""
        print("[audio_gen] Starting experiment...")
        
        rng = np.random.RandomState(42)
        
        for waveform_type in self.waveform_types:
            print(f"[audio_gen] Processing: {waveform_type}")
            
            self.results[waveform_type] = {}
            
            # Generate audio
            audio = SyntheticAudio(
                duration_seconds=self.duration,
                sample_rate=self.sample_rate,
                waveform_type=waveform_type,
                frequency=440.0,
                seed=42,
            )
            
            train_bytes, test_bytes = audio.train_test_split(test_ratio=0.2)
            
            # Calculate period in samples (for periodic prediction)
            period_samples = int(self.sample_rate / 440.0)  # ~18 samples at 8kHz
            
            print(f"[audio_gen] Train: {len(train_bytes)}, Test: {len(test_bytes)}, "
                  f"Period: {period_samples} samples")
            
            # Train manifold
            tokenizer_config = TokenizerConfig(
                hash_vocab_size=4096,
                hash_prime=31,
                segment_size=period_samples,  # Align with period
            )
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=False,
                    generator=lambda tb=train_bytes: (bytes([b]) for b in tb),
                    geometric=GeometricSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                    ),
                    spectral=SpectralSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                    ),
                    tokenizer=tokenizer_config,
                    position_init="random",
                    position_init_seed=42,
                ),
                observers={
                    "spectral": InferenceObserver([CarrierObserver(None)])
                }
            )
            
            state = manifold.run()
            
            # Set up dual-domain inference
            geo_state = {
                "positions": state.get("positions"),
                "velocities": state.get("velocities"),
                "energies": state.get("energies"),
                "heats": state.get("heats"),
                "excitations": state.get("excitations"),
                "token_ids": state.get("token_ids"),
                "masses": state.get("masses"),
            }
            carriers = manifold.carriers or {}
            
            inpainter = AudioInpainter(
                vocab_size=4096, prime=31, period_samples=period_samples
            )
            inpainter.learn_from_manifold(geo_state, carriers)
            
            # Test at different mask levels
            for mask_frac in self.mask_fracs:
                n_mask = int(len(test_bytes) * mask_frac)
                mask_positions = list(rng.choice(len(test_bytes), size=n_mask, replace=False))
                
                # Apply mask
                masked = bytearray(test_bytes)
                for pos in mask_positions:
                    masked[pos] = 128  # Mid-level as mask
                
                # Inpaint
                reconstructed = inpainter.inpaint(bytes(masked), mask_positions)
                
                # Calculate metrics
                original_np = dequantize_bytes_to_audio(test_bytes)
                recon_np = dequantize_bytes_to_audio(reconstructed)
                
                mae = np.mean(np.abs(original_np - recon_np))
                mse = np.mean((original_np - recon_np) ** 2)
                
                # SNR (Signal-to-Noise Ratio)
                signal_power = np.mean(original_np ** 2)
                noise_power = mse
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                
                self.results[waveform_type][mask_frac] = {
                    "mae": float(mae),
                    "mse": float(mse),
                    "snr": float(snr),
                }
                
                # Save one example per waveform type
                if mask_frac == 0.2 and not any(
                    e["waveform_type"] == waveform_type for e in self.examples
                ):
                    self.examples.append({
                        "waveform_type": waveform_type,
                        "original": test_bytes,
                        "masked": bytes(masked),
                        "reconstructed": reconstructed,
                        "mask_frac": mask_frac,
                    })
                
                print(f"[audio_gen] {waveform_type} @ {mask_frac*100:.0f}%: "
                      f"MAE={mae:.4f}, SNR={snr:.1f}dB")
        
        self.observe(state)
        print("[audio_gen] Experiment complete.")