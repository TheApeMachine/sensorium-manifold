"""MNIST trie recall experiment (paper-ready).

This experiment uses the clean composable pattern:
- Datasets: FilesystemDataset for MNIST train and holdout
- Observers: DehashObserver + EnergyObserver for reconstruction
- Projectors: ReconstructionProjector for image output

Produces:
- `artifacts/mnist_recall.png`
"""

from __future__ import annotations

from pathlib import Path

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import FilesystemDataset, FilesystemConfig

# MNIST image constants
MNIST_IMAGE_SIZE = 28 * 28  # 784 bytes per image

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.modes import ModeObserver
from sensorium.observers.energy import EnergyObserver
from sensorium.observers.metrics import DehashObserver

# 3. MANIFOLD
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    CoherenceSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig

# 4. PROJECTORS
from sensorium.projectors import PipelineProjector, ConsoleProjector
from sensorium.projectors.reconstruction import ReconstructionProjector, ReconstructionConfig


class MNISTTrieRecallExperiment(Experiment):
    """MNIST holdout recall from a compressed thermodynamic trie.
    
    Clean pattern:
    - datasets: FilesystemDataset for train and holdout MNIST
    - manifold: Runs simulation
    - inference: DehashObserver + EnergyObserver for reconstruction
    - projector: ReconstructionProjector for image output
    """

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        self.data_dir = self.repo_root / "data" / "mnist"
        
        # 1. DATASETS
        self.train_dataset = FilesystemDataset(FilesystemConfig(
            path=self.data_dir / "MNIST" / "raw" / "train-images-idx3-ubyte",
            header_size=16,
            limit=100,
            segment_size=MNIST_IMAGE_SIZE,
        ))
        
        self.holdout_dataset = FilesystemDataset(FilesystemConfig(
            path=self.data_dir / "MNIST" / "raw" / "t10k-images-idx3-ubyte",
            header_size=16,
            offset=1000,
            limit=5,
            segment_size=MNIST_IMAGE_SIZE,
        ))
        
        # 2. OBSERVERS
        self.inference = InferenceObserver(
            DehashObserver(
                prime=31,
                vocab=4096,
                segment_size=MNIST_IMAGE_SIZE,
            ),
        )
        
        # 3. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            ReconstructionProjector(
                config=ReconstructionConfig(
                    name="mnist_recall",
                    output_type="image",
                    field="reconstructed",
                    image_size=(28, 28),
                    colormap="gray",
                ),
                output_dir=Path("artifacts"),
            ),
        )

    def run(self):
        """Run MNIST trie recall experiment."""
        print("[mnist_trie_recall] Starting experiment...")
        
        # 2. MANIFOLD
        manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                generator=None,
                geometric=GeometricSimulationConfig(grid_size=(32, 32, 32), dt=0.01),
                coherence=CoherenceSimulationConfig(grid_size=(32, 32, 32), dt=0.01),
                tokenizer=TokenizerConfig(
                    hash_vocab_size=4096,
                    hash_prime=31,
                ),
            ),
            observers={
                "coherence": InferenceObserver(ModeObserver()),
            },
        )
        
        # Run on train data first
        print("[mnist_trie_recall] Training on train set...")
        manifold.set_generator(self.train_dataset.generate)
        manifold.run(settle=False, inference=False)
        
        # Then on holdout for inference
        print("[mnist_trie_recall] Running inference on holdout...")
        manifold.set_generator(self.holdout_dataset.generate)
        state = manifold.run(settle=True, inference=True)
        
        # 3. OBSERVE - delegate to InferenceObserver
        observation = self.inference.observe({
            "token_ids": state.get("token_ids"),
            "energies": state.get("energies"),
        })
        
        # Add energy-based reconstruction
        dehash_result = observation.as_dict()
        prompt_flat = dehash_result.get("prompt_flat")
        energy_by_tid = dehash_result.get("energy_by_tid")
        
        if prompt_flat is not None and energy_by_tid is not None:
            energy_observer = EnergyObserver(
                prime=31,
                vocab=4096,
                MNIST_IMAGE_SIZE=MNIST_IMAGE_SIZE,
            )
            energy_observer.prompt_flat = prompt_flat
            energy_observer.prompt_len = MNIST_IMAGE_SIZE
            energy_observer.energy_by_tid = energy_by_tid
            
            reconstructed = energy_observer.observe()
            self.inference.observe({}, reconstructed=reconstructed)
            
            print(f"  Reconstructed image shape: {reconstructed.shape}")
        
        # 4. PROJECT
        result = self.projector.project(self.inference)
        
        print("[mnist_trie_recall] Experiment complete.")
        return result

    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass

    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
