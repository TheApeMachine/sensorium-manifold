"""Image Generation Experiment

Uses MNIST / CIFAR-10 from HuggingFace.
Standard image benchmarks.

Goal: Generate images via thermodynamic diffusion in frequency space.
Metrics: Reconstruction MSE, FID (if we can compute it)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.spectral.unified import UnifiedManifold, Modality

from .base import BaseExperiment, Scale


class ImageGenerationExperiment(BaseExperiment):
    """Image generation using thermodynamic dynamics in frequency space.
    
    The approach:
    1. Encode training images to frequency-space particles
    2. Build attractors from the frequency distributions
    3. For generation: seed with noise, let particles diffuse toward attractors
    4. Decode via inverse FFT
    """
    
    name = "image_gen"
    goal = "Generate images via thermodynamic diffusion in frequency space"
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        super().__init__(scale, device, seed)
        
        # Scale-specific configs
        if scale == Scale.TOY:
            self.dataset_name = "mnist"
            self.image_size = 28
            self.top_k_freq = 50
        elif scale == Scale.MEDIUM:
            self.dataset_name = "mnist"
            self.image_size = 28
            self.top_k_freq = 200
        else:
            self.dataset_name = "cifar10"
            self.image_size = 32
            self.top_k_freq = 500
        
        # Frequency statistics (learned from training data)
        self._freq_attractors: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    
    def setup(self) -> None:
        """Load image dataset and initialize model."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"    Loading {self.dataset_name}...")
        
        if self.dataset_name == "mnist":
            dataset = load_dataset(
                "ylecun/mnist",
                streaming=True,
                trust_remote_code=True,
            )
            self.image_key = "image"
        else:  # cifar10
            dataset = load_dataset(
                "uoft-cs/cifar10",
                streaming=True,
                trust_remote_code=True,
            )
            self.image_key = "img"
        
        # Store iterators
        self.train_stream = dataset["train"]
        self.eval_stream = dataset["test"]
        
        # Prefetch images
        self._train_images: List[torch.Tensor] = []
        self._eval_images: List[torch.Tensor] = []
        self._train_labels: List[int] = []
        self._eval_labels: List[int] = []
        
        self._load_images(
            self.train_stream,
            self._train_images,
            self._train_labels,
            self.scale_config.max_train_samples or 1000,
        )
        self._load_images(
            self.eval_stream,
            self._eval_images,
            self._eval_labels,
            self.scale_config.max_eval_samples or 200,
        )
        
        print(f"    Train images: {len(self._train_images)}")
        print(f"    Eval images: {len(self._eval_images)}")
        print(f"    Image size: {self.image_size}x{self.image_size}")
        
        # Initialize manifold
        self.manifold = UnifiedManifold(
            self.physics_config,
            self.device,
            embed_dim=self.scale_config.embed_dim,
        )
    
    def _load_images(
        self,
        stream,
        images: List[torch.Tensor],
        labels: List[int],
        max_samples: int,
    ) -> None:
        """Load images from stream."""
        for sample in stream:
            img = sample[self.image_key]
            
            # Convert PIL to tensor if needed
            if hasattr(img, "convert"):
                img = img.convert("L")  # Grayscale
                import numpy as np
                img = np.array(img, dtype=np.float32) / 255.0
                img = torch.from_numpy(img)
            else:
                img = torch.tensor(img, dtype=torch.float32)
                if img.max() > 1.0:
                    img = img / 255.0
                if img.ndim == 3:
                    img = img.mean(dim=-1)  # Grayscale
            
            # Resize if needed
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0).unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            
            images.append(img)
            labels.append(sample.get("label", 0))
            
            if len(images) >= max_samples:
                break
    
    def train_iterator(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """Iterate over training images."""
        for img, label in zip(self._train_images, self._train_labels):
            yield img, label
    
    def train_step(self, batch: Tuple[torch.Tensor, int]) -> Dict[str, float]:
        """One step of thermodynamic image learning.
        
        Training = building attractor statistics from images.
        """
        image, label = batch
        image = image.to(self.device)
        
        # Clear and encode image
        self.manifold.clear()
        self.manifold.encode_image(image, top_k=self.top_k_freq)
        
        # Run dynamics to let structure emerge
        for _ in range(5):
            self.manifold.step()
        
        # Store frequency attractors (building a generative model)
        for p in self.manifold._particles:
            if p.modality == Modality.IMAGE and p.position.numel() == 2:
                u = int(p.position[0].item())
                v = int(p.position[1].item())
                key = (u, v)
                
                if key not in self._freq_attractors:
                    self._freq_attractors[key] = []
                
                # Store (energy, phase) for this frequency
                phase = p.phase[0].item() if p.phase is not None else 0.0
                self._freq_attractors[key].append((p.energy.item(), phase))
        
        # Decode and compute reconstruction error
        reconstructed = self.manifold.decode_image((self.image_size, self.image_size))
        mse = ((image - reconstructed) ** 2).mean().item()
        
        return {"mse": mse}
    
    def _generate_image(self) -> torch.Tensor:
        """Generate a new image using learned attractors."""
        self.manifold.clear()
        
        if not self._freq_attractors:
            # No attractors learned yet, return noise
            return torch.randn(self.image_size, self.image_size, device=self.device)
        
        # Create particles from attractor statistics
        for (u, v), stats in self._freq_attractors.items():
            if not stats:
                continue
            
            # Use mean energy and phase
            mean_energy = sum(e for e, _ in stats) / len(stats)
            mean_phase = sum(p for _, p in stats) / len(stats)
            
            # Add some noise for generation diversity
            noise = torch.randn(1, device=self.device).item() * 0.1
            
            position = torch.tensor([float(u), float(v)], device=self.device)
            phase = torch.tensor([mean_phase + noise], device=self.device)
            
            self.manifold.add_particle(
                position=position,
                energy=max(0.001, mean_energy + noise * 0.1),
                modality=Modality.IMAGE,
                phase=phase,
            )
        
        # Run dynamics
        for _ in range(10):
            self.manifold.step()
        
        # Decode
        return self.manifold.decode_image((self.image_size, self.image_size))
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate reconstruction and generation quality."""
        # Reconstruction quality on eval set
        mse_total = 0.0
        count = 0
        
        for image in self._eval_images[:50]:
            image = image.to(self.device)
            
            self.manifold.clear()
            self.manifold.encode_image(image, top_k=self.top_k_freq)
            
            for _ in range(5):
                self.manifold.step()
            
            reconstructed = self.manifold.decode_image((self.image_size, self.image_size))
            mse_total += ((image - reconstructed) ** 2).mean().item()
            count += 1
        
        recon_mse = mse_total / max(count, 1)
        
        # Generate some samples and compute metrics
        gen_images = [self._generate_image() for _ in range(10)]
        
        # Basic generation metrics
        gen_mean = torch.stack(gen_images).mean().item()
        gen_std = torch.stack(gen_images).std().item()
        
        # Compare to training distribution
        train_mean = torch.stack(self._train_images[:100]).mean().item()
        train_std = torch.stack(self._train_images[:100]).std().item()
        
        mean_diff = abs(gen_mean - train_mean)
        std_diff = abs(gen_std - train_std)
        
        return {
            "reconstruction_mse": recon_mse,
            "gen_mean": gen_mean,
            "gen_std": gen_std,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "num_attractors": len(self._freq_attractors),
            "eval_images": count,
        }
    
    def save_samples(self, out_dir: str, num_samples: int = 16) -> None:
        """Save generated samples for visual inspection."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from pathlib import Path
            
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Generate samples
            samples = [self._generate_image().cpu().numpy() for _ in range(num_samples)]
            
            # Plot grid
            rows = int(num_samples ** 0.5)
            cols = (num_samples + rows - 1) // rows
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten() if num_samples > 1 else [axes]
            
            for ax, sample in zip(axes, samples):
                ax.imshow(sample, cmap='gray')
                ax.axis('off')
            
            for ax in axes[len(samples):]:
                ax.axis('off')
            
            fig.tight_layout()
            fig.savefig(out_path / "generated_samples.png", dpi=150)
            plt.close(fig)
            
            print(f"    Saved samples to {out_path / 'generated_samples.png'}")
            
        except ImportError:
            print("    (matplotlib not available, skipping sample saving)")


def run_image_gen_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = ImageGenerationExperiment(scale=scale, device=device)
    result = exp.run()
    
    # Save samples
    exp.save_samples("./artifacts/image_gen")
    
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
    }
