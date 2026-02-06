"""Collision experiment demonstrating hash collisions as compression.

This experiment demonstrates the clean composable pattern:
- Datasets: SyntheticDataset with COLLISION pattern
- Observers: Composable metric observers
- Projectors: Config-driven outputs that query InferenceObserver

The experiment is pure orchestration - no inline data processing.
"""

from __future__ import annotations

from sensorium.experiments.base import (
    Experiment, 
)
from sensorium.manifold import Manifold
from sensorium.observers.sql import SQLObserver
from sensorium.tokenizer.universal import UniversalTokenizer
from sensorium.dataset import (
    HuggingFaceDataset, HuggingFaceConfig,
)
from sensorium.observers.inference import InferenceObserver


class CollisionExperiment(Experiment):
    """Demonstrate Thermodynamic Trie (Collision is Compression)

    To avoid further confusion, when we talk about "collision" in this experiment,
    we mean physical collision of particles, no hash collisions.
    In all fairness, we did indeed have a hash collision issue, however that was a spur
    of the moment idea that we were in no way dependent on. And so, we have since just
    moved to still using raw bytes, just with the sequence index weaved in.
    Problem solved. Collision is Compression, yet again (but kinda always was anyway).
    """

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(
            experiment_name, 
            profile, 
            dashboard=dashboard, 
            reportable=[
                "n_tokens",
                "grid_size",
                "max_modes",
                "max_steps",
                "num_units",
                "unit_length",
                "seed",
                "collision_rate",
                "collision_rate_observed",
                "n_unique_tokens",
                "compression_ratio",
                "key_collision_rate",
                "unique_keys",
                "fold_top1",
                "fold_pr",
                "transition_top1_prob",
                "transition_unique_edges",
                "spatial_clustering",
                "entropy",
                "mode_participation",
                "mode_entropy",
                "psi_delta_rel"
            ]
        )

        self.datasets = [HuggingFaceDataset(HuggingFaceConfig(
            name="nyu-mll/glue",
            subset="mrpc",
            split="train",
            field="text",
            streaming=True,
        ))]

        self.manifold = Manifold(
            tokenizer=UniversalTokenizer(
                datasets=self.datasets
            )
        )
        
    def run(self):
        """Run experiment for each collision rate."""
        self.project(self.observe(self.manifold.run()))
        
    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        return InferenceObserver(
            SQLObserver("""
                SELECT 
                    CAST(particles AS TEXT), 
                    load_count 
                FROM simulation;
            """)),
    
    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        return self.projector.project(self.inference)
