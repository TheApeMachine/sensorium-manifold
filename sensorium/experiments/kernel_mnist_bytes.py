"""Kernel MNIST bytes: vectorized manifold ingest.

This experiment uses the clean composable pattern:
- Datasets: FilesystemDataset with MNIST binary data
- Observers: InferenceObserver with particle/carrier metrics
- Projectors: Console output for real-time feedback

Each byte becomes one "token" in the carrier dynamics. Position resets
every 784 bytes (one image), so bytes at the same pixel position across
different images hash consistently.
"""

from __future__ import annotations

from pathlib import Path

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig
from optimizer.tokenizer import TokenizerConfig

# Datasets
from sensorium.dataset import FilesystemDataset, FilesystemConfig

# Observers
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    ParticleCount,
    ModeCount,
    MeanParticleEnergy,
    CompressionRatio,
)

# Projectors
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
    LaTeXTableProjector,
    TableConfig,
)


# MNIST binary format constants
IMAGE_HEADER_SIZE = 16  # magic(4) + count(4) + rows(4) + cols(4)
IMAGE_SIZE = 28 * 28    # 784 bytes per image


class KernelMNISTBytes(Experiment):
    """MNIST vectorized ingest experiment.
    
    Clean pattern:
    - datasets: FilesystemDataset for MNIST binary
    - manifold: Runs simulation
    - inference: Composes metric observers, accumulates results
    - projector: Console output for feedback
    """

    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # 1. DATASET
        mnist_path = self.repo_root / "data" / "mnist" / "train-images-idx3-ubyte"
        
        self.dataset = FilesystemDataset(FilesystemConfig(
            path=mnist_path,
            header_size=IMAGE_HEADER_SIZE,
            segment_size=IMAGE_SIZE,  # Reset index every 784 bytes (one image)
        ))
        
        # 2. MANIFOLD
        self.manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                tokenizer=TokenizerConfig(
                    hash_vocab_size=4096,
                    hash_prime=31,
                ),
            )
        )
        
        # 3. OBSERVERS - composable metrics
        self.inference = InferenceObserver([
            ParticleCount(),
            ModeCount(),
            MeanParticleEnergy(),
            CompressionRatio(),
        ])
        
        # 4. PROJECTORS - console feedback
        self.projector = PipelineProjector(
            ConsoleProjector(
                fields=["n_particles", "n_modes", "mean_particle_energy", "compression_ratio"],
                format="table",
            ),
            LaTeXTableProjector(
                TableConfig(
                    name="mnist_bytes_summary",
                    columns=["n_particles", "n_modes", "compression_ratio"],
                    headers={
                        "n_particles": "Particles",
                        "n_modes": "Modes",
                        "compression_ratio": "Compression",
                    },
                    caption="MNIST bytes ingest summary",
                    label="tab:mnist_bytes",
                ),
                output_dir=self.artifact_path("tables"),
            ),
        )

    def run(self):
        """Run vectorized ingest and observe results."""
        print("[mnist_bytes] Starting experiment...")
        
        # Run simulation with dataset
        self.manifold.add_dataset(self.dataset.generate)
        state = self.manifold.run()
        
        # Observe - InferenceObserver accumulates results
        self.inference.observe(
            state,
            manifold=self.manifold,
            experiment="mnist_bytes",
        )
        
        # Project all results
        self.project()
        
        print("[mnist_bytes] Experiment complete.")

    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        return self.inference.observe(state, manifold=self.manifold)

    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        return self.projector.project(self.inference)
