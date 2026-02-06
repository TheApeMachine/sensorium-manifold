"""Cross-modal experiment: Text and Image in unified manifold.

This experiment uses the clean composable pattern:
- Datasets: CrossModalDataset (image as frequency bytes + text labels)
- Observers: CrossModalDynamicsObserver, ImageReconstructor, ParticleCount, ModeCount
- Projectors: CrossModalFigureProjector, CrossModalTableProjector

Produces:
- `paper/figures/cross_modal.png` - Multi-panel hero figure
- `paper/figures/cross_modal_embedding.png` - 3D embedding space
- `paper/figures/frequency_particles.png` - 2D frequency distribution
- `paper/tables/cross_modal_summary.tex` - Metrics table
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    CrossModalConfig,
    CrossModalDataset,
    create_stripe_image,
    create_checkerboard_image,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    CrossModalDynamicsObserver,
    ImageReconstructor,
    ParticleCount,
    ModeCount,
)

# 3. MANIFOLD
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    CoherenceSimulationConfig,
    GeometricSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig

# 4. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.cross_modal import (
    CrossModalFigureConfig,
    CrossModalFigureProjector,
    CrossModalTableProjector,
)


class KernelCrossModal(Experiment):
    """Cross-modal experiment demonstrating unified text-image processing.
    
    Clean pattern:
    - datasets: CrossModalDataset for each test case
    - manifold: Runs simulation
    - inference: InferenceObserver with dynamics and reconstruction observers
    - projector: CrossModalFigureProjector, CrossModalTableProjector
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        self.vocab_size = 4096
        self.prime = 31
        self.image_size = 32
        self.top_k_freq = 64
        self.n_steps = 150
        
        # 2. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 4. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            CrossModalFigureProjector(
                config=CrossModalFigureConfig(name="cross_modal"),
                output_dir=Path("paper/figures"),
            ),
            CrossModalTableProjector(output_dir=Path("paper/tables")),
        )

    def run(self):
        print("[cross_modal] Starting cross-modal experiment...")
        
        test_cases = [
            ("horizontal", create_stripe_image(self.image_size, "horizontal"), ["horizontal", "stripes", "lines"]),
            ("vertical", create_stripe_image(self.image_size, "vertical"), ["vertical", "stripes", "lines"]),
            ("diagonal", create_stripe_image(self.image_size, "diagonal"), ["diagonal", "stripes", "pattern"]),
            ("checkerboard", create_checkerboard_image(self.image_size), ["checkerboard", "grid", "pattern"]),
        ]
        
        tests = []
        reconstructor = ImageReconstructor()
        
        for name, image, text_labels in test_cases:
            print(f"\n  Processing: {name}")
            
            # 1. DATASET
            dataset = CrossModalDataset(
                image=image,
                text_labels=text_labels,
                config=CrossModalConfig(top_k_freq=self.top_k_freq),
            )
            
            # 2. MANIFOLD
            dynamics_observer = CrossModalDynamicsObserver(max_steps=self.n_steps)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(
                        grid_size=(32, 32, 32),
                    ),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=(32, 32, 32),
                        stable_amp_threshold=0.15,
                        crystallize_amp_threshold=0.20,
                    ),
                ),
                observers={"coherence": dynamics_observer}
            )
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            # 3. OBSERVE - reconstruct image
            recon_result = reconstructor.observe({
                "state": state,
                "metadata": dataset.metadata,
                "n_image_bytes": len(dataset.image_bytes),
                "original": image,
            })
            
            # Extract frequency data for visualization
            freq_data = dataset.get_frequency_space()
            
            test_result = {
                "name": name,
                "original": image,
                "reconstructed": recon_result["reconstructed"],
                "mse": recon_result["mse"],
                "psnr": recon_result["psnr"],
                "wall_time_ms": wall_time,
                "n_particles": len(dataset.combined_data),
                "n_image_particles": len(dataset.image_bytes),
                "n_text_particles": len(dataset.text_bytes),
                "text_labels": text_labels,
                "history": dynamics_observer.history,
                "freq_data": freq_data,
            }
            tests.append(test_result)
            
            print(f"    MSE: {recon_result['mse']:.4f}, PSNR: {recon_result['psnr']:.2f} dB")
        
        # Accumulate to inference observer
        self.inference.observe(
            {},
            tests=tests,
        )
        
        # Project
        self.project()
        
        print("\n[cross_modal] Experiment complete.")
    
    def observe(self, state: dict) -> dict:
        """Observer interface for compatibility."""
        return {}
    
    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
