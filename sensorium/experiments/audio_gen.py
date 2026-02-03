"""Audio/Music Generation Experiment

Uses NSynth or similar audio datasets from HuggingFace.
Tests spectral manifold's audio synthesis capabilities.

Goal: Generate audio via thermodynamic diffusion in frequency space.
Metrics: Reconstruction MSE, Spectral distance
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.spectral.unified import UnifiedManifold, Modality

from .base import BaseExperiment, Scale


class AudioGenerationExperiment(BaseExperiment):
    """Audio generation using thermodynamic dynamics in frequency space.
    
    The approach:
    1. Encode audio samples to frequency-space particles via FFT
    2. Build attractors from frequency distributions
    3. For generation: seed with noise, let particles diffuse
    4. Decode via inverse FFT
    """
    
    name = "audio_gen"
    goal = "Generate audio via thermodynamic diffusion in frequency space"
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        super().__init__(scale, device, seed)
        
        # Scale-specific configs
        if scale == Scale.TOY:
            self.sample_rate = 16000
            self.audio_length = 8000  # 0.5 seconds
            self.top_k_freq = 50
            self.max_samples = 100
        elif scale == Scale.MEDIUM:
            self.sample_rate = 16000
            self.audio_length = 16000  # 1 second
            self.top_k_freq = 200
            self.max_samples = 1000
        else:
            self.sample_rate = 22050
            self.audio_length = 44100  # 2 seconds
            self.top_k_freq = 500
            self.max_samples = 10000
        
        # Frequency statistics (learned from training data)
        self._freq_attractors: Dict[int, List[Tuple[float, float]]] = {}
    
    def setup(self) -> None:
        """Load audio dataset and initialize model."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"    Loading audio dataset (speech_commands)...")
        
        # Use Google Speech Commands as a simple audio dataset
        # It's smaller and easier to work with than NSynth
        try:
            dataset = load_dataset(
                "google/speech_commands",
                "v0.02",
                streaming=True,
            )
            self.audio_key = "audio"
            self.has_real_data = True
        except Exception as e:
            print(f"    Could not load speech_commands: {e}")
            print(f"    Falling back to synthetic audio")
            self.has_real_data = False
            dataset = None
        
        if self.has_real_data and dataset is not None:
            self.train_stream = dataset["train"]
            self.eval_stream = dataset["validation"]
            
            # Prefetch audio samples
            self._train_audio: List[torch.Tensor] = []
            self._eval_audio: List[torch.Tensor] = []
            
            self._load_audio(
                self.train_stream,
                self._train_audio,
                min(self.scale_config.max_train_samples or 1000, self.max_samples),
            )
            self._load_audio(
                self.eval_stream,
                self._eval_audio,
                min(self.scale_config.max_eval_samples or 200, self.max_samples // 5),
            )
        else:
            # Generate synthetic audio for testing
            self._train_audio = [self._generate_synthetic() for _ in range(100)]
            self._eval_audio = [self._generate_synthetic() for _ in range(20)]
        
        print(f"    Train samples: {len(self._train_audio)}")
        print(f"    Eval samples: {len(self._eval_audio)}")
        print(f"    Sample rate: {self.sample_rate}")
        print(f"    Audio length: {self.audio_length} samples ({self.audio_length/self.sample_rate:.2f}s)")
        
        # Initialize manifold
        self.manifold = UnifiedManifold(
            self.physics_config,
            self.device,
            embed_dim=self.scale_config.embed_dim,
        )
    
    def _generate_synthetic(self) -> torch.Tensor:
        """Generate synthetic audio for testing when real data unavailable."""
        t = torch.linspace(0, self.audio_length / self.sample_rate, self.audio_length)
        
        # Random combination of sine waves
        freq1 = 220 + torch.randint(0, 440, (1,)).item()  # A3-A4 range
        freq2 = freq1 * 2  # Octave
        freq3 = freq1 * 1.5  # Fifth
        
        audio = (
            0.5 * torch.sin(2 * torch.pi * freq1 * t) +
            0.3 * torch.sin(2 * torch.pi * freq2 * t) +
            0.2 * torch.sin(2 * torch.pi * freq3 * t)
        )
        
        # Add envelope
        envelope = torch.exp(-3 * t / (self.audio_length / self.sample_rate))
        audio = audio * envelope
        
        return audio
    
    def _load_audio(
        self,
        stream,
        output: List[torch.Tensor],
        max_samples: int,
    ) -> None:
        """Load audio from stream."""
        for sample in stream:
            audio_data = sample[self.audio_key]
            
            # Extract array from audio dict
            if isinstance(audio_data, dict):
                arr = audio_data.get("array", [])
                sr = audio_data.get("sampling_rate", self.sample_rate)
            else:
                arr = audio_data
                sr = self.sample_rate
            
            # Convert to tensor
            audio = torch.tensor(arr, dtype=torch.float32)
            
            # Resample if needed
            if sr != self.sample_rate:
                # Simple resampling (not ideal but works for demo)
                ratio = self.sample_rate / sr
                new_len = int(len(audio) * ratio)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=new_len,
                    mode='linear',
                    align_corners=False,
                ).squeeze()
            
            # Pad or truncate to fixed length
            if len(audio) < self.audio_length:
                audio = torch.nn.functional.pad(audio, (0, self.audio_length - len(audio)))
            else:
                audio = audio[:self.audio_length]
            
            # Normalize
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
            
            output.append(audio)
            
            if len(output) >= max_samples:
                break
    
    def train_iterator(self) -> Iterator[torch.Tensor]:
        """Iterate over training audio."""
        for audio in self._train_audio:
            yield audio
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """One step of thermodynamic audio learning."""
        audio = batch.to(self.device)
        
        # Clear and encode audio
        self.manifold.clear()
        self.manifold.encode_audio(audio, sample_rate=self.sample_rate, top_k=self.top_k_freq)
        
        # Run dynamics
        for _ in range(5):
            self.manifold.step()
        
        # Store frequency attractors
        for p in self.manifold._particles:
            if p.modality == Modality.AUDIO and p.position.numel() == 1:
                freq_bin = int(p.position[0].item())
                
                if freq_bin not in self._freq_attractors:
                    self._freq_attractors[freq_bin] = []
                
                phase = p.phase[0].item() if p.phase is not None else 0.0
                self._freq_attractors[freq_bin].append((p.energy.item(), phase))
        
        # Decode and compute reconstruction error
        reconstructed = self.manifold.decode_audio(self.audio_length, self.sample_rate)
        mse = ((audio - reconstructed) ** 2).mean().item()
        
        return {"mse": mse}
    
    def _generate_audio(self) -> torch.Tensor:
        """Generate new audio using learned attractors."""
        self.manifold.clear()
        
        if not self._freq_attractors:
            return torch.randn(self.audio_length, device=self.device)
        
        # Create particles from attractor statistics
        for freq_bin, stats in self._freq_attractors.items():
            if not stats:
                continue
            
            mean_energy = sum(e for e, _ in stats) / len(stats)
            mean_phase = sum(p for _, p in stats) / len(stats)
            
            noise = torch.randn(1, device=self.device).item() * 0.1
            
            position = torch.tensor([float(freq_bin)], device=self.device)
            phase = torch.tensor([mean_phase + noise], device=self.device)
            
            self.manifold.add_particle(
                position=position,
                energy=max(0.001, mean_energy + noise * 0.1),
                modality=Modality.AUDIO,
                phase=phase,
            )
        
        # Run dynamics
        for _ in range(10):
            self.manifold.step()
        
        return self.manifold.decode_audio(self.audio_length, self.sample_rate)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate reconstruction and generation quality."""
        mse_total = 0.0
        spectral_dist_total = 0.0
        count = 0
        
        for audio in self._eval_audio[:20]:
            audio = audio.to(self.device)
            
            self.manifold.clear()
            self.manifold.encode_audio(audio, sample_rate=self.sample_rate, top_k=self.top_k_freq)
            
            for _ in range(5):
                self.manifold.step()
            
            reconstructed = self.manifold.decode_audio(self.audio_length, self.sample_rate)
            
            # Time-domain MSE
            mse_total += ((audio - reconstructed) ** 2).mean().item()
            
            # Spectral distance
            orig_spec = torch.fft.rfft(audio).abs()
            recon_spec = torch.fft.rfft(reconstructed).abs()
            spectral_dist_total += ((orig_spec - recon_spec) ** 2).mean().item()
            
            count += 1
        
        recon_mse = mse_total / max(count, 1)
        spectral_dist = spectral_dist_total / max(count, 1)
        
        # Generate some samples
        gen_audio = [self._generate_audio() for _ in range(5)]
        
        # Basic generation metrics
        gen_energy = torch.stack(gen_audio).abs().mean().item()
        
        return {
            "reconstruction_mse": recon_mse,
            "spectral_distance": spectral_dist,
            "gen_energy": gen_energy,
            "num_freq_attractors": len(self._freq_attractors),
            "eval_samples": count,
        }
    
    def save_samples(self, out_dir: str, num_samples: int = 4) -> None:
        """Save generated audio samples."""
        try:
            import scipy.io.wavfile as wav
            from pathlib import Path
            import numpy as np
            
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_samples):
                audio = self._generate_audio().cpu().numpy()
                audio = np.clip(audio, -1, 1)
                audio = (audio * 32767).astype(np.int16)
                
                wav.write(
                    str(out_path / f"generated_{i}.wav"),
                    self.sample_rate,
                    audio,
                )
            
            print(f"    Saved {num_samples} audio samples to {out_path}")
            
        except ImportError:
            print("    (scipy not available, skipping audio saving)")


def run_audio_gen_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = AudioGenerationExperiment(scale=scale, device=device)
    result = exp.run()
    
    # Save samples
    exp.save_samples("./artifacts/audio_gen")
    
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
    }
