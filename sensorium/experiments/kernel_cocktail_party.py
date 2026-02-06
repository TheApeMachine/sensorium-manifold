"""Kernel cocktail party (multiple-speaker) separation experiment.

This experiment uses the clean composable pattern:
- Datasets: CocktailPartyDataset (STFT-based audio processing)
- Observers: SpectralClusteringObserver, ParticleCount, ModeCount
- Projectors: CocktailPartyTableProjector, CocktailPartyFigureProjector

Produces:
- `paper/tables/cocktail_party_summary.tex`
- `paper/figures/cocktail_party.png`
- `artifacts/speaker_0.wav`, `artifacts/speaker_1.wav`
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    CocktailPartyConfig,
    CocktailPartyDataset,
    save_wav,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    SpectralClusteringObserver,
    ParticleCount,
    ModeCount,
)

# 3. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.cocktail_party import (
    CocktailPartyFigureConfig,
    CocktailPartyTableProjector,
    CocktailPartyFigureProjector,
)


class KernelCocktailParty(Experiment):
    """Cocktail party separation using spectral manifold dynamics.
    
    Clean pattern:
    - datasets: CocktailPartyDataset for audio loading and STFT
    - observers: SpectralClusteringObserver for speaker clustering
    - projector: CocktailPartyTableProjector, CocktailPartyFigureProjector
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        self.num_speakers = 2
        
        # 2. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 3. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            CocktailPartyTableProjector(output_dir=Path("paper/tables")),
            CocktailPartyFigureProjector(
                config=CocktailPartyFigureConfig(name="cocktail_party"),
                output_dir=Path("paper/figures"),
            ),
        )
        
    def run(self):
        print(f"[cocktail_party] Starting experiment...")
        
        # 1. DATASET
        wav_path = Path(__file__).parent / "two_speakers.wav"
        dataset = CocktailPartyDataset(CocktailPartyConfig(wav_path=wav_path))
        
        if dataset.audio_samples is None:
            print(f"[cocktail_party] ERROR: Could not load audio")
            return
        
        # Prepare frequency data
        magnitudes = dataset.active_magnitudes
        freq_normalized = dataset.bin_indices / dataset.n_bins
        energy_normalized = magnitudes / (magnitudes.max() + 1e-10)
        
        # 2. OBSERVE - cluster frequencies
        start_time = time.time()
        
        clustering = SpectralClusteringObserver(num_speakers=self.num_speakers)
        cluster_result = clustering.observe({
            "frequencies": freq_normalized,
            "energies": energy_normalized,
        })
        
        wall_time_ms = (time.time() - start_time) * 1000
        
        labels = cluster_result.get("labels")
        
        if labels is None:
            print(f"[cocktail_party] ERROR: Clustering failed")
            return
        
        # Print cluster stats
        for i in range(self.num_speakers):
            count = cluster_result["cluster_counts"][i]
            mean_freq = cluster_result["cluster_means"][i]
            print(f"[cocktail_party] Cluster {i}: {count:,} bins, mean_freq={mean_freq:.3f}")
        print(f"[cocktail_party] Separation score: {cluster_result['separation_score']:.2f}")
        
        # 3. Reconstruct separated audio
        print(f"[cocktail_party] Reconstructing separated audio...")
        separated_audio = dataset.reconstruct_separated_audio(labels, self.num_speakers)
        
        # Save separated audio
        for speaker_id, speaker_audio in enumerate(separated_audio):
            out_path = Path("artifacts") / f"speaker_{speaker_id}.wav"
            save_wav(speaker_audio, out_path, sample_rate=dataset.config.sample_rate)
            duration = len(speaker_audio) / dataset.config.sample_rate
            print(f"  Saved {out_path} ({duration:.2f}s)")
        
        # Accumulate to inference observer
        self.inference.observe(
            {},
            labels=labels,
            cluster_counts=cluster_result["cluster_counts"],
            cluster_means=cluster_result["cluster_means"],
            cluster_energy=cluster_result["cluster_energy"],
            separation_score=cluster_result["separation_score"],
            freq_normalized=freq_normalized,
            energy_normalized=energy_normalized,
            frame_indices=dataset.frame_indices,
            bin_indices=dataset.bin_indices,
            n_particles=dataset.n_particles,
            n_frames=dataset.n_frames,
            n_bins=dataset.n_bins,
        )
        
        # Project
        self.project()
        
        # Write simulation stats
        self.write_simulation_stats(
            "cocktail_party",
            n_particles=dataset.n_particles,
            n_modes=0,
            n_crystallized=0,
            grid_size=(64, 64, 64),
            dt=0.01,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/cocktail_party_stats.tex")
        
        print(f"[cocktail_party] Experiment complete.")
    
    def observe(self, state: dict) -> dict:
        """Observer interface for compatibility."""
        return {}
    
    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
