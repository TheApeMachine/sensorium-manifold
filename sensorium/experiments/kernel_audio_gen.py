"""Kernel audio waveform inpainting via thermodynamic trie.

This experiment uses the clean composable pattern:
- Datasets: AudioDataset (quantizes audio to bytes)
- Observers: AudioPeriodicityPredictor, ParticleCount, ModeCount
- Projectors: AudioTableProjector, AudioFigureProjector

Produces:
- `paper/tables/audio_gen_summary.tex`
- `paper/figures/audio_gen.png`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    AudioDatasetConfig,
    AudioDataset,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    AudioPeriodicityPredictor,
    ParticleCount,
    ModeCount,
)

# 3. MANIFOLD
from optimizer.manifold import Manifold, SimulationConfig, CoherenceSimulationConfig
from optimizer.tokenizer import TokenizerConfig

# 4. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.audio import (
    AudioTableProjector,
    AudioFigureProjector,
    AudioFigureConfig,
)


class KernelAudioGen(Experiment):
    """Audio waveform inpainting experiment.
    
    Clean pattern:
    - datasets: AudioDataset for each waveform type
    - manifold: Runs simulation
    - inference: InferenceObserver with AudioPeriodicityPredictor
    - projector: AudioTableProjector, AudioFigureProjector
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Configuration
        self.vocab_size = 4096
        self.prime = 31
        self.sample_rate = 8000
        self.duration = 1.0
        self.frequency = 440.0
        self.waveform_types = ["sine", "square", "sawtooth", "mixed"]
        self.mask_fracs = [0.1, 0.2, 0.3]
        
        # 3. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 4. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            AudioTableProjector(output_dir=Path("paper/tables")),
            AudioFigureProjector(
                config=AudioFigureConfig(
                    name="audio_gen",
                    sample_rate=self.sample_rate,
                ),
                output_dir=Path("paper/figures"),
            ),
        )
        
        self._stats = None

    def run(self):
        """Run the audio generation experiment."""
        import time
        
        print("[audio_gen] Starting experiment...")
        
        rng = np.random.RandomState(42)
        period_samples = int(self.sample_rate / self.frequency)
        print(f"[audio_gen] Period: {period_samples} samples at {self.frequency} Hz")
        
        # Create predictor observer
        predictor = AudioPeriodicityPredictor(period_samples=period_samples)
        
        for waveform_type in self.waveform_types:
            print(f"[audio_gen] Processing: {waveform_type}")
            
            # 1. DATASET
            dataset = AudioDataset(AudioDatasetConfig(
                waveform_type=waveform_type,
                sample_rate=self.sample_rate,
                duration=self.duration,
                frequency=self.frequency,
                seed=42,
            ))
            
            print(f"[audio_gen] Train: {len(dataset.train_bytes)}, Test: {len(dataset.test_bytes)}")
            
            # 2. MANIFOLD
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        stable_amp_threshold=0.15,
                        crystallize_amp_threshold=0.20,
                        grid_size=(64, 64, 64),
                        dt=0.01,
                    ),
                )
            )
            
            start_time = time.time()
            state = manifold.run()
            run_time_ms = (time.time() - start_time) * 1000
            
            # Capture stats from first waveform
            if self._stats is None:
                self._stats = {
                    "grid_size": (64, 64, 64),
                    "dt": 0.01,
                    "wall_time_ms": run_time_ms,
                }
            
            # 3. OBSERVE - test at different mask levels
            for mask_frac in self.mask_fracs:
                n_mask = int(len(dataset.test_bytes) * mask_frac)
                mask_positions = set(rng.choice(len(dataset.test_bytes), size=n_mask, replace=False))
                
                # Run predictor
                prediction = predictor.observe({
                    "train_bytes": dataset.train_bytes,
                    "test_bytes": dataset.test_bytes,
                    "split_idx": dataset.split_idx,
                    "mask_positions": mask_positions,
                })
                
                # Accumulate to inference observer with metadata
                save_example = (mask_frac == 0.2 and waveform_type == "sine")
                
                self.inference.observe(
                    state,
                    manifold=manifold,
                    waveform_type=waveform_type,
                    mask_frac=mask_frac,
                    mae=prediction.get("mae", 0),
                    mse=prediction.get("mse", 0),
                    snr=prediction.get("snr", 0),
                    accuracy=prediction.get("accuracy", 0),
                    n_masked=prediction.get("n_masked", 0),
                    n_correct=prediction.get("n_correct", 0),
                    # For figure panel A
                    original_bytes=dataset.test_bytes if save_example else None,
                    reconstructed_bytes=prediction.get("reconstructed") if save_example else None,
                )
                
                print(f"[audio_gen] {waveform_type} @ {mask_frac*100:.0f}%: "
                      f"MAE={prediction.get('mae', 0):.4f}, SNR={prediction.get('snr', 0):.1f}dB, "
                      f"Acc={prediction.get('accuracy', 0):.1%}")
        
        # Project
        self.project()
        
        # Write simulation stats
        if self._stats:
            results = self.inference.results
            if results:
                self.write_simulation_stats(
                    "audio_gen",
                    n_particles=results[0].get("n_particles", 0),
                    n_modes=results[0].get("n_modes", 0),
                    n_crystallized=0,
                    grid_size=self._stats["grid_size"],
                    dt=self._stats["dt"],
                    n_steps=1,
                    wall_time_ms=self._stats["wall_time_ms"],
                )
                print("✓ Generated: paper/tables/audio_gen_stats.tex")
        
        print("[audio_gen] Experiment complete.")

    def project(self):
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
