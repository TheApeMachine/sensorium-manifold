"""Kernel image handling via Universal Tokenizer (MNIST inpainting demo).

This experiment uses the clean composable pattern:
- Datasets: MNISTDataset for train/test images
- Observers: ImageInpainter, ParticleCount, ModeCount
- Projectors: ImageTableProjector, ImageFigureProjector

Produces:
- `paper/tables/image_gen_summary.tex`
- `paper/figures/image_gen.png`
"""

from __future__ import annotations

import time
from pathlib import Path

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    MNISTConfig,
    MNISTDataset,
    MNIST_IMAGE_SIZE,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    ImageInpainter,
    ParticleCount,
    ModeCount,
)
from sensorium.observers.modes import ModeObserver

# 3. MANIFOLD
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    CoherenceSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig

# 4. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.image import (
    ImageFigureConfig,
    ImageTableProjector,
    ImageFigureProjector,
)


class KernelImageGen(Experiment):
    """MNIST inpainting experiment using Universal Tokenizer.
    
    Clean pattern:
    - datasets: MNISTDataset for train/test
    - manifold: Runs simulation on training data
    - inference: InferenceObserver with ImageInpainter
    - projector: ImageTableProjector, ImageFigureProjector
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        self.train_images = 100
        self.test_images = 20
        self.hash_vocab_size = 4096
        self.mask_fracs = [0.1, 0.2, 0.3, 0.5]
        
        # 2. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 3. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            ImageTableProjector(
                output_dir=Path("paper/tables"),
                train_images=self.train_images,
                test_images=self.test_images,
            ),
            ImageFigureProjector(
                config=ImageFigureConfig(name="image_gen"),
                output_dir=Path("paper/figures"),
            ),
        )

    def run(self):
        """Run MNIST inpainting experiment."""
        print("[image_gen] Starting experiment...")
        
        # 1. DATASETS
        data_dir = self.repo_root / "data" / "mnist"
        train_dataset = MNISTDataset(MNISTConfig(
            data_dir=data_dir,
            train=True,
            limit=self.train_images,
        ))
        test_dataset = MNISTDataset(MNISTConfig(
            data_dir=data_dir,
            train=False,
            limit=self.test_images,
        ))
        
        train_images = train_dataset.images
        test_images = test_dataset.images
        test_labels = test_dataset.labels
        
        print(f"[image_gen] Train: {len(train_images)}, Test: {len(test_images)}")
        
        # 2. MANIFOLD - train on training images
        grid_size = (32, 32, 32)
        dt = 0.01
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                generator=train_dataset.generate,
                geometric=GeometricSimulationConfig(
                    grid_size=grid_size,
                    dt=dt,
                ),
                coherence=CoherenceSimulationConfig(
                    grid_size=grid_size,
                    dt=dt,
                ),
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.hash_vocab_size,
                    hash_prime=31,
                ),
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "coherence": InferenceObserver([ModeObserver()])
            }
        )
        
        start_time = time.time()
        state = manifold.run()
        wall_time_ms = (time.time() - start_time) * 1000
        
        # Set up dual-domain inference
        geo_state = {
            "positions": state.get("positions"),
            "velocities": state.get("velocities"),
            "energies": state.get("energies"),
            "heats": state.get("heats"),
            "excitations": state.get("excitations"),
            "token_ids": state.get("token_ids"),
            "masses": state.get("masses"),
        }
        modes = manifold.modes or {}
        
        inpainter = ImageInpainter(
            vocab_size=self.hash_vocab_size,
            prime=31,
            image_size=MNIST_IMAGE_SIZE,
        )
        inpainter.learn_from_manifold(geo_state, modes)
        
        # 3. OBSERVE - batch evaluation at different mask levels
        print(f"[image_gen] Testing mask fractions: {self.mask_fracs}")
        batch_result = inpainter.evaluate_batch(
            test_images=list(test_images),
            test_labels=list(test_labels),
            mask_fracs=self.mask_fracs,
            seed=42,
        )
        
        mask_results = batch_result["mask_results"]
        examples = batch_result["examples"]
        
        for mask_frac, metrics in mask_results.items():
            print(f"[image_gen] Mask {mask_frac*100:.0f}%: "
                  f"MAE={metrics['mae']:.2f}, PSNR={metrics['psnr']:.2f}dB")
        
        # Get carrier stats
        amplitudes = modes.get("amplitudes")
        n_modes = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
        crystallized = modes.get("crystallized")
        n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
        n_particles = len(geo_state["token_ids"]) if geo_state.get("token_ids") is not None else 0
        
        # Accumulate to inference observer
        self.inference.observe(
            state,
            manifold=manifold,
            mask_results=mask_results,
            examples=examples,
        )
        
        # Project
        self.project()
        
        # Write simulation stats
        self.write_simulation_stats(
            "image_gen",
            n_particles=n_particles,
            n_modes=n_modes,
            n_crystallized=n_crystallized,
            grid_size=grid_size,
            dt=dt,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/image_gen_stats.tex")
        
        print("[image_gen] Experiment complete.")

    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass

    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
