"""Example experiment template demonstrating the clean composable pattern.

This file shows the ideal structure for experiments:
1. Datasets - Composable data sources (SyntheticDataset, FilesystemDataset, etc.)
2. Manifold - Simulation engine, fed datasets via add_dataset()
3. Observers - Composable measurements via InferenceObserver
4. Projectors - Composable outputs (tables, figures, reconstructions)

The experiment orchestrates these components without any inline data generation.
"""

from __future__ import annotations

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig
from optimizer.tokenizer import TokenizerConfig

# Datasets
from sensorium.dataset import SyntheticDataset, SyntheticConfig, SyntheticPattern

# Observers  
from sensorium.observers import observe_reaction, Particles, Modes
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import SpatialClustering, CompressionRatio

# Projectors
from sensorium.projectors import PipelineProjector, LaTeXTableProjector, FigureProjector


class ExampleExperiment(Experiment):
    """Example experiment showing the clean composable pattern."""
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # 1. DATASETS - composable data sources
        self.datasets = [
            SyntheticDataset(SyntheticConfig(
                pattern=SyntheticPattern.COLLISION,
                num_units=10,
                unit_length=32,
                collision_rate=0.5,
            ))
        ]
        
        # 2. MANIFOLD - simulation engine
        self.manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                tokenizer=TokenizerConfig(),
            ),
            observers={
                "geometric": observe_reaction(Particles()),
                "coherence": observe_reaction(Modes()),
            }
        )
        
        # 3. OBSERVERS - composable measurements
        self.inference = InferenceObserver([
            SpatialClustering(),
            CompressionRatio(),
        ])
        
        # 4. PROJECTORS - composable outputs
        self.projector = PipelineProjector(
            LaTeXTableProjector(output_dir=self.artifact_path("tables")),
            FigureProjector(output_dir=self.artifact_path("figures")),
        )

    def run(self):
        """Run the experiment."""
        for dataset in self.datasets:
            self.manifold.add_dataset(dataset.generate)
            state = self.manifold.run()
        
        # Observe -> Project
        observation = self.observe(state)
        self.project(observation)

    def observe(self, state: dict) -> dict:
        """Observe the manifold state."""
        return self.inference.observe(state)

    def project(self, observation: dict) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(observation)
