"""MNIST Raw Bytes Classification Experiment

Demonstrates true modality-agnostic learning: classify MNIST digits
using raw pixel bytes as tokens. No encoder, no spectral decomposition—
just a stream of 784 bytes per image.

The manifold treats images as "text in a 256-character alphabet" and
learns byte-level transition patterns that distinguish digit classes.

Key claim: The same dynamics that learn language can learn vision,
because the manifold doesn't know what modality it's processing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.semantic.manifold import SemanticManifold
from thermo_manifold.semantic.hierarchical import HierarchicalSemanticManifold

from .base import BaseExperiment, ExperimentResult, Scale, ScaleConfig

DEFAULT_OUT_DIR = Path("./artifacts/mnist_bytes")

# Custom scale configs for MNIST (different from text experiments)
# Note: embed_dim should be <= vocab_size for proper embeddings
MNIST_SCALE_CONFIGS = {
    Scale.TOY: ScaleConfig(
        name="toy",
        max_train_samples=1000,
        max_eval_samples=200,
        embed_dim=512,  # Smaller than hash_vocab_size (4096) + 10 labels
        train_steps=1000,
        eval_every=200,
        dt=0.02,
    ),
    Scale.MEDIUM: ScaleConfig(
        name="medium",
        max_train_samples=10000,
        max_eval_samples=1000,
        embed_dim=512,
        train_steps=10000,
        eval_every=1000,
        dt=0.02,
    ),
    Scale.FULL: ScaleConfig(
        name="full",
        max_train_samples=60000,
        max_eval_samples=10000,
        embed_dim=512,
        train_steps=60000,
        eval_every=5000,
        dt=0.02,
    ),
}


class MNISTBytesExperiment(BaseExperiment):
    """MNIST classification using raw pixel bytes with position-aware hashing.
    
    Each image is a sequence of 784 bytes (28x28 pixels, 0-255).
    Each (byte_value, position) pair is hashed to a unique token ID,
    preserving spatial information while treating the image as a byte stream.
    
    The label is appended as a special token after the image bytes.
    
    Training: Feed [hashed_bytes..., label_token], learn transitions.
    Inference: Feed [hashed_bytes...], predict which label has highest flow.
    """
    
    name = "mnist_bytes"
    goal = "Classify MNIST digits from raw pixel bytes (no encoder)"
    
    # Hash parameters for position-aware byte tokens
    # We use a prime-based hash to mix byte value and position
    HASH_PRIME = 31
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
        context_window: int = 64,  # How many recent bytes to use as context
        dashboard: bool = False,  # Enable real-time dashboard
        ponder: bool = False,  # Enable expensive idle pondering
        hash_vocab_size: int = 4096,  # Size of hashed token vocabulary
    ):
        super().__init__(scale, device, seed)
        
        # Override scale config with MNIST-specific settings
        self.scale_config = MNIST_SCALE_CONFIGS[scale]
        
        # Vocab: hash_vocab_size position-aware byte tokens + 10 label tokens
        self.hash_vocab_size = hash_vocab_size
        self.num_labels = 10
        self.label_offset = hash_vocab_size  # Labels are tokens hash_vocab_size to hash_vocab_size+9
        self.vocab_size = hash_vocab_size + self.num_labels
        
        # Context window (can't use all 784 bytes as context—too long)
        self.context_window = context_window
        
        # Dashboard
        self.use_dashboard = dashboard
        self._dashboard = None
        self.enable_ponder = ponder
        
        # Build vocabulary (for SemanticManifold compatibility)
        self.vocab = [f"hash_{i}" for i in range(hash_vocab_size)] + [f"label_{i}" for i in range(10)]
        
        # Data storage
        self._train_images: List[torch.Tensor] = []
        self._train_labels: List[int] = []
        self._eval_images: List[torch.Tensor] = []
        self._eval_labels: List[int] = []
        
        # Manifold
        self.manifold: Optional[SemanticManifold] = None
    
    def _hash_byte_position(self, byte_value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Hash (byte_value, position) pairs to token IDs in [0, hash_vocab_size).
        
        Uses a simple but effective hash: (byte * prime + position) mod vocab_size.
        This ensures different positions with the same byte value get different tokens.
        
        Args:
            byte_value: Tensor of byte values (0-255)
            position: Tensor of position indices (0-783 for MNIST)
            
        Returns:
            Tensor of hashed token IDs in [0, hash_vocab_size)
        """
        byte_value = byte_value.to(torch.long)
        position = position.to(torch.long)
        # Hash formula: (byte * HASH_PRIME + position) mod hash_vocab_size
        hashed = (byte_value * self.HASH_PRIME + position) % self.hash_vocab_size
        return hashed
    
    def setup(self) -> None:
        """Load MNIST and initialize manifold."""
        print("    Loading MNIST...")
        
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("Please install torchvision: pip install torchvision")
        
        # Load MNIST (downloads if needed)
        data_dir = Path("./data/mnist")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = datasets.MNIST(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        
        eval_dataset = datasets.MNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        
        # Convert to byte tensors (0-255)
        max_train = self.scale_config.max_train_samples or len(train_dataset)
        max_eval = self.scale_config.max_eval_samples or len(eval_dataset)
        
        print(f"    Processing {max_train} training images...")
        for i in range(min(max_train, len(train_dataset))):
            img, label = train_dataset[i]
            # img is (1, 28, 28) float in [0, 1] -> convert to bytes
            img_bytes = (img * 255).to(torch.uint8).flatten()
            self._train_images.append(img_bytes)
            self._train_labels.append(int(label))
        
        print(f"    Processing {max_eval} evaluation images...")
        for i in range(min(max_eval, len(eval_dataset))):
            img, label = eval_dataset[i]
            img_bytes = (img * 255).to(torch.uint8).flatten()
            self._eval_images.append(img_bytes)
            self._eval_labels.append(int(label))
        
        print(f"    Train images: {len(self._train_images)}")
        print(f"    Eval images: {len(self._eval_images)}")
        print(f"    Vocabulary: {self.vocab_size} tokens ({self.hash_vocab_size} hashed + 10 labels)")
        print(f"    Context window: {self.context_window} bytes")
        print(f"    Position-aware hashing: enabled (prime={self.HASH_PRIME})")
        
        # Initialize hierarchical semantic manifold (with chunks)
        self.manifold = HierarchicalSemanticManifold(
            config=self.physics_config,
            device=self.device,
            vocab=self.vocab,
            embed_dim=self.scale_config.embed_dim,
            chunk_min_len=2,  # Minimum chunk length (bigrams)
            chunk_max_len=4,  # Maximum chunk length (4-grams)
        )
    
    def _image_to_sequence(self, img_bytes: torch.Tensor, label: int) -> torch.Tensor:
        """Convert image bytes + label to a token sequence.
        
        Returns: tensor of shape (785,) with byte tokens followed by label token.
        """
        # Image bytes are already 0-255, which maps directly to token IDs 0-255
        img_tokens = img_bytes.to(torch.long)
        
        # Label token is 256 + label
        label_token = torch.tensor([self.label_offset + label], dtype=torch.long)
        
        # Concatenate
        sequence = torch.cat([img_tokens, label_token])
        return sequence
    
    def train_iterator(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """Iterate over training images.
        
        Yields: (image_bytes_tensor, label)
        """
        indices = torch.randperm(len(self._train_images)).tolist()
        for idx in indices:
            yield self._train_images[idx], self._train_labels[idx]
    
    def train_step(self, batch: Tuple[torch.Tensor, int]) -> Dict[str, float]:
        """One step of training: stream full image as position-hashed byte sequence, then observe label.
        
        Each (byte_value, position) pair is hashed to a unique token, preserving spatial info.
        The manifold learns sequential transitions throughout the image,
        building up a representation that associates position-aware byte patterns with labels.
        
        Returns dict with standard metrics:
        - loss: cross-entropy loss (same as surprise)
        - ppl: perplexity = exp(loss)
        - entropy: entropy of prediction distribution
        """
        img_bytes, label = batch
        
        # Create position indices for all bytes
        positions = torch.arange(len(img_bytes), device=self.device, dtype=torch.long)
        byte_values = img_bytes.to(torch.long).to(self.device)
        
        # Hash (byte, position) pairs to get position-aware tokens
        img_tokens = self._hash_byte_position(byte_values, positions)
        
        # Stream the full image as a sequence of hashed tokens
        # Process in chunks to build up context incrementally
        chunk_size = self.context_window
        n_chunks = (len(img_tokens) + chunk_size - 1) // chunk_size
        
        # Accumulate per-token losses for average
        token_losses = []
        
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(img_tokens))
            chunk = img_tokens[start:end]
            
            # Ingest this chunk
            self.manifold.ingest_ids(chunk)
            
            # Run grammar step to process transitions within the chunk
            self.manifold.step_grammar()
            
            # Observe token-to-token transitions within the chunk
            for j in range(len(chunk) - 1):
                obs = self.manifold.observe_next_token(
                    int(chunk[j + 1].item()),
                    cur_id=int(chunk[j].item()),
                )
                # surprise = -log(P(next|cur)) = cross-entropy loss per token
                token_losses.append(obs.get("surprise", 0.0))
        
        # After streaming the full image, observe the label transition
        # The label follows the last hashed token of the image
        # Use predict_from_token on the LAST token specifically (not the whole context)
        last_token = int(img_tokens[-1].item())
        probs = self.manifold.predict_from_token(last_token)
        probs_sum = probs.sum()
        if probs_sum > self.manifold.cfg.eps:
            probs = probs / probs_sum
        else:
            # No outgoing edges yet - use uniform over labels
            probs = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
            for i in range(self.num_labels):
                probs[self.label_offset + i] = 1.0 / self.num_labels
        
        label_token = self.label_offset + label
        label_prob = float(probs[label_token].detach().item())
        obs = self.manifold.observe_next_token(label_token, probs=probs, cur_id=last_token)
        
        # The label prediction loss is the most important one
        label_loss = obs.get("surprise", 0.0)
        token_losses.append(label_loss)
        
        # Compute standard metrics
        # Loss: average cross-entropy across all tokens (including label)
        avg_loss = sum(token_losses) / len(token_losses) if token_losses else 0.0
        
        # Perplexity: exp(loss) - how "surprised" the model is on average
        # Clamp to avoid overflow
        import math
        ppl = math.exp(min(avg_loss, 20.0))  # Cap at exp(20) ≈ 485 million
        
        # Entropy of current prediction distribution
        entropy = float(self.manifold.entropy().item())
        
        # Return with standard metric names
        result = {
            "loss": avg_loss,
            "ppl": ppl,
            "entropy": entropy,
            "label_loss": label_loss,  # Loss specifically on label prediction
            "label_prob": label_prob,
            **obs,  # Include original observation metrics
        }
        
        return result
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate classification accuracy on eval set."""
        correct = 0
        total = 0
        
        # Per-class accuracy tracking
        class_correct = [0] * 10
        class_total = [0] * 10
        
        for img_bytes, label in zip(self._eval_images, self._eval_labels):
            # Create position indices and hash (byte, position) pairs
            positions = torch.arange(len(img_bytes), device=self.device, dtype=torch.long)
            byte_values = img_bytes.to(torch.long).to(self.device)
            img_tokens = self._hash_byte_position(byte_values, positions)
            
            # Reset excitation state for fresh prediction on each sample
            self.manifold.attractors.set(
                "excitation",
                torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32),
            )
            
            # Stream the full image as a sequence of hashed tokens (same as training)
            chunk_size = self.context_window
            n_chunks = (len(img_tokens) + chunk_size - 1) // chunk_size
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(img_tokens))
                chunk = img_tokens[start:end]
                
                self.manifold.ingest_ids(chunk)
                self.manifold.step_grammar()
            
            # Run additional grammar steps to let the state settle
            for _ in range(2):
                self.manifold.step_grammar()
            
            # Predict label from the LAST token specifically
            # This is what matters: given the last pixel, what label follows?
            last_token = int(img_tokens[-1].item())
            probs = self.manifold.predict_from_token(last_token)
            probs_sum = probs.sum()
            if probs_sum > self.manifold.cfg.eps:
                probs = probs / probs_sum
            
            # Get label predictions
            label_probs = probs[self.label_offset:self.label_offset + self.num_labels]
            predicted_label = int(torch.argmax(label_probs).item())
            
            if predicted_label == label:
                correct += 1
                class_correct[label] += 1
            
            total += 1
            class_total[label] += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-class accuracies
        per_class = {}
        for i in range(10):
            if class_total[i] > 0:
                per_class[f"acc_digit_{i}"] = class_correct[i] / class_total[i]
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "graph_edges": self.manifold.graph.num_edges,
            **per_class,
        }
    
    def run(self) -> ExperimentResult:
        """Run the MNIST bytes experiment with custom artifact generation."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.name} ({self.scale.value})")
        print(f"Goal: {self.goal}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        
        try:
            print("\n[1] Setup...")
            self.setup()
            
            out_dir = DEFAULT_OUT_DIR / self.scale.value
            out_dir.mkdir(parents=True, exist_ok=True)
            
            steps = self.scale_config.train_steps
            print(f"\n[2] Training ({steps} steps)...")
            
            data_iter = self.train_iterator()
            
            # Initialize dashboard if requested
            if self.use_dashboard:
                from ..core.dashboard import Dashboard, SimpleDashboard
                import sys
                if sys.platform == "darwin":
                    # macOS GUI backends must run on the main thread.
                    self._dashboard = SimpleDashboard(self.manifold, vocab=self.vocab)
                    self._dashboard_mode = "simple"
                    print("    Dashboard: using simple mode on macOS")
                else:
                    try:
                        self._dashboard = Dashboard(self.manifold, vocab=self.vocab)
                        self._dashboard.start()
                        self._dashboard_mode = "threaded"
                    except Exception as e:
                        print(f"    Dashboard init failed ({e}), using simple mode")
                        self._dashboard = SimpleDashboard(self.manifold, vocab=self.vocab)
                        self._dashboard_mode = "simple"
            
            if tqdm is not None:
                pbar = tqdm(range(steps), desc="Training", unit="step")
            else:
                pbar = range(steps)
            
            for t in pbar:
                self.state.step = t
                
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.state.epoch += 1
                    data_iter = self.train_iterator()
                    batch = next(data_iter)
                
                metrics = self.train_step(batch)
                
                # Update dashboard every 10 steps (for debugging)
                if self._dashboard is not None and t % 10 == 0:
                    if self._dashboard_mode == "threaded":
                        self._dashboard.update(step=t, extra=metrics)
                    else:
                        self._dashboard.update_and_render(step=t, extra=metrics)
                
                # Occasional stochastic traversal (but not too often—it's expensive)
                if self.enable_ponder and t > 0 and t % 500 == 0:
                    self.manifold.idle_think(steps=1, dream_steps=2)
            
            if tqdm is not None and hasattr(pbar, 'close'):
                pbar.close()
            
            # One final stochastic traversal to consolidate
            if self.enable_ponder:
                print("    Running final stochastic traversal...")
                self.manifold.idle_think(steps=2, dream_steps=4)
            
            print("\n[3] Final evaluation...")
            final_metrics = self.evaluate()
            
            print(f"\n    {'─'*40}")
            print(f"    Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"    Correct: {final_metrics['correct']} / {final_metrics['total']}")
            print(f"    Graph edges: {final_metrics['graph_edges']}")
            print(f"    {'─'*40}")
            
            # Save metrics
            metrics = {
                "config": asdict(self.physics_config),
                "scale": self.scale.value,
                "final_eval": final_metrics,
            }
            
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str),
                encoding="utf-8",
            )
            
            # Generate plots
            self._generate_plots(out_dir, final_metrics)
            
            # Close dashboard and generate 3D graph
            if self._dashboard is not None:
                if self._dashboard_mode == "threaded":
                    self._dashboard.save_snapshot(str(out_dir / "dashboard_final.png"))
                    self._dashboard.stop()
                else:
                    self._dashboard.save(str(out_dir / "dashboard_final.png"))
                    # Generate 3D bond graph visualization
                    try:
                        self._dashboard.render_3d_graph(
                            save_path=str(out_dir / "bond_graph_3d.png"),
                            max_nodes=300,
                            max_edges=1000,
                        )
                    except Exception as e:
                        print(f"    (3D graph generation failed: {e})")
                    self._dashboard.close()
            
            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=True,
                metrics={"final": final_metrics},
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"\n[FAILED] {e}")
            print(tb)
            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=False,
                metrics={},
                failure_reason=str(e),
            )
    
    def _generate_plots(
        self,
        out_dir: Path,
        final_metrics: Dict[str, Any],
    ) -> None:
        """Generate visualization plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("    (matplotlib not available, skipping plots)")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Per-class accuracy bar chart
        class_accs = [final_metrics.get(f"acc_digit_{i}", 0) for i in range(10)]
        bars = ax.bar(range(10), class_accs, color='steelblue')
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Random baseline (10%)')
        ax.axhline(y=final_metrics['accuracy'], color='red', linestyle='-', alpha=0.7, label=f'Overall: {final_metrics["accuracy"]:.1%}')
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"MNIST Classification from Raw Bytes\n(No encoder, {final_metrics['graph_edges']} edges learned)")
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(out_dir / "mnist_bytes.png", dpi=150)
        plt.close(fig)
        
        print(f"    Saved plot to {out_dir / 'mnist_bytes.png'}")


def run_mnist_bytes_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = MNISTBytesExperiment(scale=scale, device=device)
    result = exp.run()
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MNIST Raw Bytes Classification")
    parser.add_argument(
        "--scale",
        type=str,
        default="toy",
        choices=["toy", "medium", "full"],
        help="Experiment scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable real-time dashboard visualization",
    )
    parser.add_argument(
        "--ponder",
        action="store_true",
        help="Enable expensive idle pondering steps",
    )
    
    args = parser.parse_args()
    
    scale = Scale(args.scale)
    device = torch.device(args.device) if args.device else None
    
    exp = MNISTBytesExperiment(
        scale=scale,
        device=device,
        dashboard=args.dashboard,
        ponder=args.ponder,
    )
    result = exp.run()
    if result.success:
        print("\n✓ Experiment completed successfully")
    else:
        print("\n✗ Experiment failed")
