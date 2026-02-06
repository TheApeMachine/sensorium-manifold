"""Kernel text "diffusion" (byte denoising) via thermodynamic trie.

This experiment uses the clean composable pattern:
- Datasets: DiffusionDataset (repetitive text patterns)
- Observers: InferenceObserver with TriePatternMatcher
- Projectors: Config-driven tables and figures

Produces:
- `paper/tables/text_diffusion_summary.tex`
- `paper/figures/text_diffusion.png`
"""

from __future__ import annotations

from typing import Set

import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, CoherenceSimulationConfig
from optimizer.tokenizer import TokenizerConfig

# Datasets
from sensorium.dataset import DiffusionDataset, DiffusionDatasetConfig

# Observers
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    TriePatternMatcher,
    ParticleCount,
    ModeCount,
)

# Projectors
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
    DiffusionTableProjector,
    DiffusionFigureProjector,
    DiffusionFigureConfig,
)


class KernelTextDiffusion(Experiment):
    """Text byte denoising/inpainting experiment.
    
    Clean pattern:
    - datasets: DiffusionDataset with repetitive text
    - manifold: Runs simulation
    - inference: InferenceObserver accumulates results across mask levels
    - projector: Config-driven table and figure projectors
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Configuration
        self.vocab_size = 4096
        self.prime = 31
        self.context_length = 5
        self.mask_fracs = [0.1, 0.2, 0.3, 0.5]
        
        # 1. DATASET
        self.dataset = DiffusionDataset(DiffusionDatasetConfig(
            max_bytes=2000,
            segment_size=64,
            seed=42,
        ))
        
        # 2. MANIFOLD - configured in run()
        self.manifold = None
        
        # 3. OBSERVERS
        self.pattern_matcher = TriePatternMatcher(
            vocab_size=self.vocab_size,
            prime=self.prime,
            context_length=self.context_length,
            segment_size=self.dataset.segment_size,
        )
        
        self.inference = InferenceObserver([
            ParticleCount(),
            ModeCount(),
        ])
        
        # 4. PROJECTORS - config-driven
        self.projector = PipelineProjector(
            ConsoleProjector(
                fields=["mask_frac", "char_accuracy", "n_correct", "n_masked"],
                format="table",
            ),
            DiffusionTableProjector(
                output_dir=self.artifact_path("tables"),
            ),
            DiffusionFigureProjector(
                DiffusionFigureConfig(name="text_diffusion"),
                output_dir=self.artifact_path("figures"),
            ),
        )

    def run(self):
        """Run the text diffusion experiment."""
        print("[text_diffusion] Starting experiment...")
        
        train_bytes = self.dataset.train_bytes
        test_bytes = self.dataset.test_bytes
        
        print(f"[text_diffusion] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
        
        # 2. MANIFOLD
        self.manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
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
        
        # Run simulation once
        self.manifold.add_dataset(self.dataset.generate)
        state = self.manifold.run()
        
        token_ids = state.get("token_ids")
        if token_ids is None:
            print("[text_diffusion] ERROR: No token_ids")
            return
        
        energies = state.get("energies", torch.ones(len(token_ids)))
        
        # Test at different mask levels
        rng = np.random.RandomState(42)
        
        for mask_frac in self.mask_fracs:
            print(f"\n{'='*60}")
            print(f"Testing mask fraction: {mask_frac*100:.0f}%")
            print('='*60)
            
            # Generate random mask positions
            n_mask = int(len(test_bytes) * mask_frac)
            mask_positions: Set[int] = set(rng.choice(len(test_bytes), size=n_mask, replace=False))
            
            # Apply pattern matcher
            result = self.pattern_matcher.observe({
                "train_bytes": train_bytes,
                "test_bytes": test_bytes,
                "token_ids": token_ids,
                "energies": energies,
                "mask_positions": mask_positions,
            })
            
            # Calculate hamming distance
            reconstructed = result.get("reconstructed", test_bytes)
            hamming = sum(1 for a, b in zip(reconstructed, test_bytes) if a != b)
            
            # Observe - InferenceObserver accumulates results
            self.inference.observe(
                state,
                manifold=self.manifold,
                mask_frac=mask_frac,
                char_accuracy=result.get("char_accuracy", 0.0),
                n_correct=result.get("n_correct", 0),
                n_masked=result.get("n_masked", 0),
                hamming_dist=hamming,
            )
            
            print(f"  Accuracy: {result.get('char_accuracy', 0):.1%} "
                  f"({result.get('n_correct', 0)}/{result.get('n_masked', 0)})")
        
        # Project all accumulated results
        self.project()
        
        print("\n[text_diffusion] Experiment complete.")

    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        return self.inference.observe(state, manifold=self.manifold)

    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        return self.projector.project(self.inference)
