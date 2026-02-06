"""Kernel next-token (byte) prediction via Universal Tokenizer.

This experiment uses the clean composable pattern:
- Datasets: TextDataset (wraps SyntheticDataset with TEXT_PREFIX pattern)
- Observers: InferenceObserver with NextTokenMetrics
- Projectors: Config-driven tables and custom figure projector

Writes paper artifacts:
- `paper/tables/next_token_summary.tex`
- `paper/figures/next_token.png`
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    CoherenceSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig

# Datasets
from sensorium.dataset import HuggingFaceConfig, HuggingFaceDataset, TextDataset, TextDatasetConfig

# Observers
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.dual_domain import DualDomainInference
from sensorium.observers.metrics import (
    CrystallizationObserver,
    NextTokenMetrics,
    ParticleCount,
    ModeCount,
    CrystallizedCount,
)

# Projectors
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
    LaTeXTableProjector,
    TableConfig,
    NextTokenFigureProjector,
    NextTokenFigureConfig,
)


class KernelNextToken(Experiment):
    """Next-byte prediction experiment using dual-domain inference.
    
    Clean pattern:
    - datasets: TextDataset with shared prefix patterns
    - manifold: Runs simulation with crystallization observer
    - inference: InferenceObserver accumulates prediction results
    - projector: Config-driven table and custom figure projector
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Configuration
        self.context_length = 8
        self.vocab_size = 4096
        self.prime = 31
        
        # 1. DATASET
        self.dataset = HuggingFaceDataset(HuggingFaceConfig(
            name="LTCB/enwik8",
            split="train",
            field="text",
            streaming=True,
        ))

        self.dataset = TextDataset(TextDatasetConfig(
            segment_size=16,
            seed=42,
        ))
        
        # 2. MANIFOLD - configured in run() after dataset is ready
        self.manifold = None
        
        # 3. OBSERVERS
        self.crystallization = CrystallizationObserver(
            min_modes=10,
            min_crystallized=5,
            max_steps=500,
        )
        
        self.inference = InferenceObserver([
            NextTokenMetrics(),
            ParticleCount(),
            ModeCount(),
            CrystallizedCount(),
        ])
        
        # 4. PROJECTORS - config-driven
        self.projector = PipelineProjector(
            ConsoleProjector(
                fields=["accuracy", "top3_accuracy", "perplexity", "num_modes"],
                format="table",
            ),
            LaTeXTableProjector(
                TableConfig(
                    name="next_token_summary",
                    columns=[
                        "accuracy", "top3_accuracy", "top5_accuracy", 
                        "perplexity", "total_predictions",
                        "num_modes", "num_crystallized",
                    ],
                    headers={
                        "accuracy": "Accuracy",
                        "top3_accuracy": "Top-3 Acc",
                        "top5_accuracy": "Top-5 Acc",
                        "perplexity": "Perplexity",
                        "total_predictions": "N Predictions",
                        "num_modes": "Modes",
                        "num_crystallized": "Crystallized",
                    },
                    caption="Next-token prediction results",
                    label="tab:next_token",
                    precision=3,
                ),
                output_dir=self.artifact_path("tables"),
            ),
            NextTokenFigureProjector(
                NextTokenFigureConfig(
                    name="next_token",
                    segment_size=self.dataset.segment_size,
                ),
                output_dir=self.artifact_path("figures"),
            ),
        )

    def run(self):
        """Run the next-byte prediction experiment."""
        print("[next_token] Starting experiment...")
        
        # Get train/test data
        train_bytes, test_bytes = self.dataset.train_test_split()
        segment_size = self.dataset.segment_size
        
        print(f"[next_token] Training on {len(train_bytes)} bytes")
        print(f"[next_token] Testing on {len(test_bytes)} bytes")
        print(f"[next_token] Pattern frequencies: {self.dataset.get_pattern_stats()}")
        
        # 2. MANIFOLD
        self.manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                ),
                coherence=CoherenceSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                    max_carriers=64,
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                    volatile_decay_mul=0.98,
                    coupling_scale=0.5,
                ),
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "coherence": InferenceObserver([self.crystallization])
            }
        )
        
        # Run simulation
        self.manifold.add_dataset(self.dataset.generate)
        state = self.manifold.run()
        modes = self.manifold.modes or {}
        
        # Create dual-domain inference engine
        geo_state = {
            "positions": state.get("positions"),
            "velocities": state.get("velocities"),
            "energies": state.get("energies"),
            "heats": state.get("heats"),
            "excitations": state.get("excitations"),
            "token_ids": state.get("token_ids"),
            "masses": state.get("masses"),
        }
        
        dual_inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=modes,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
        
        print(f"[next_token] Modes formed: {dual_inference.num_modes}")
        
        crystallized = dual_inference.crystallized_modes()
        num_crystallized = crystallized.mode_indices.numel()
        print(f"[next_token] Crystallized modes: {num_crystallized}")
        
        # Run predictions
        predictions = self._run_predictions(
            dual_inference, test_bytes, train_bytes, segment_size
        )
        
        print(f"[next_token] Made {len(predictions)} predictions")
        
        # Observe - pass predictions and metadata
        self.inference.observe(
            {
                **state,
                "predictions": predictions,
                "modes": modes,
            },
            manifold=self.manifold,
            segment_size=segment_size,
            context_length=self.context_length,
            num_modes=dual_inference.num_modes,
            num_crystallized=num_crystallized,
        )
        
        # Project results
        self.project()
        
        print("[next_token] Experiment complete.")
    
    def _run_predictions(
        self,
        inference: DualDomainInference,
        test_bytes: bytes,
        train_bytes: bytes,
        segment_size: int,
    ) -> List[Dict[str, Any]]:
        """Run predictions on test data.
        
        Returns list of prediction dicts with keys:
        - position, actual, predicted, top3, top5, scores
        """
        predictions = []
        test_offset = len(train_bytes) % segment_size
        
        for idx in range(self.context_length, len(test_bytes)):
            actual = test_bytes[idx]
            context_start = idx - self.context_length
            context = test_bytes[context_start:idx]
            context_start_pos = (test_offset + context_start) % segment_size
            
            scores, _ = inference.predict_next_byte(
                context_bytes=context,
                context_start_position=context_start_pos,
                segment_size=segment_size,
                method="wave",
            )
            
            predicted = int(np.argmax(scores))
            top_indices = np.argsort(scores)[::-1]
            
            predictions.append({
                "position": idx,
                "actual": actual,
                "predicted": predicted,
                "top3": list(top_indices[:3]),
                "top5": list(top_indices[:5]),
                "scores": scores.copy(),
            })
        
        return predictions

    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        return self.inference.observe(state, manifold=self.manifold)

    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        return self.projector.project(self.inference)
