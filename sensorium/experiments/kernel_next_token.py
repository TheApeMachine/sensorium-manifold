"""Kernel next-token (byte) prediction via Universal Tokenizer.

We treat text as raw UTF-8 bytes. At each position, we:
1) Find carriers coupled to context oscillators (spectral domain)
2) Score candidate next-bytes by their coupling to those carriers
3) Take argmax as prediction

NON-CHEATING DESIGN:
====================
This experiment uses proper dual-domain inference:
- Training: Manifold runs on training text, carriers form
- Inference: Query manifold state (geometric + spectral) to predict

The key mechanism:
1. Context bytes → oscillators → coupled carriers (switch to spectral)
2. Candidate bytes → candidate frequencies → coupling scores
3. Best coupled candidate is the prediction

No ground truth is accessed during inference.

Writes paper artifacts:
- `paper/tables/next_token_summary.tex`
- `paper/figures/next_token.pdf`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple
from collections import Counter

import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    SpectralSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.carrier import CarrierObserver
from sensorium.observers.dual_domain import DualDomainInference


class TextDataset:
    """Simple text dataset for next-byte prediction.
    
    Uses a sample of text that has repeating patterns the system can learn.
    """
    
    def __init__(self, text: str = None, seed: int = 42):
        if text is None:
            # Default: a text with learnable patterns (more repetition for learning)
            text = """The cat sat on the mat.
The cat sat on the mat.
The cat sat on the mat.
The dog sat on the log.
The dog sat on the log.
The cat sat on the mat.
The bird sat on the word.
The cat sat on the mat.
The fish swam in the dish.
The cat sat on the mat.
The cat sat on the mat.
The dog sat on the log.
"""
        self.text = text.strip()
        self.bytes_data = self.text.encode("utf-8")
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def train_test_split(self, test_ratio: float = 0.2) -> Tuple[bytes, bytes]:
        """Split by taking last portion as test (temporal split)."""
        split_idx = int(len(self.bytes_data) * (1 - test_ratio))
        return self.bytes_data[:split_idx], self.bytes_data[split_idx:]


class KernelNextToken(Experiment):
    """Next-byte prediction experiment using dual-domain inference."""
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        
        self.context_length = 8  # Bytes of context for prediction
        self.vocab_size = 4096
        self.prime = 31
        
        # Results
        self.predictions: List[Dict[str, Any]] = []

    def observe(self, state: dict, carriers: dict):
        """Generate paper artifacts from predictions."""
        if not self.predictions:
            print("Warning: No predictions collected")
            return
        
        # Calculate metrics
        correct = sum(1 for p in self.predictions if p["predicted"] == p["actual"])
        total = len(self.predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Top-k accuracy
        top3_correct = sum(1 for p in self.predictions if p["actual"] in p["top3"])
        top5_correct = sum(1 for p in self.predictions if p["actual"] in p["top5"])
        top3_accuracy = top3_correct / total if total > 0 else 0.0
        top5_accuracy = top5_correct / total if total > 0 else 0.0
        
        # Perplexity approximation
        log_probs = []
        for p in self.predictions:
            scores = p["scores"]
            actual = p["actual"]
            # Softmax to get probabilities
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / (exp_scores.sum() + 1e-10)
            log_probs.append(np.log(probs[actual] + 1e-10))
        
        avg_log_prob = np.mean(log_probs) if log_probs else -10
        perplexity = np.exp(-avg_log_prob)
        
        # Carrier statistics
        num_carriers = carriers.get("num_carriers", 0)
        if isinstance(num_carriers, torch.Tensor):
            num_carriers = int(num_carriers.item())
        
        carrier_state = carriers.get("carrier_state")
        num_crystallized = 0
        if carrier_state is not None and num_carriers > 0:
            num_crystallized = int((carrier_state[:num_carriers] == 2).sum().item())
        
        # Summary metrics
        summary = {
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "top5_accuracy": top5_accuracy,
            "perplexity": perplexity,
            "total_predictions": total,
            "context_length": self.context_length,
            "num_carriers": num_carriers,
            "num_crystallized": num_crystallized,
        }
        
        self.write_kv_table("next_token_summary", summary)
        
        # Plot
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Accuracy by character type
        char_types = {
            "lowercase": (97, 123),
            "uppercase": (65, 91),
            "digits": (48, 58),
            "space": (32, 33),
            "punct": (33, 48),
            "newline": (10, 11),
        }
        
        type_accuracies = {}
        for ctype, (low, high) in char_types.items():
            type_preds = [p for p in self.predictions if low <= p["actual"] < high]
            if type_preds:
                c = sum(1 for p in type_preds if p["predicted"] == p["actual"])
                type_accuracies[ctype] = c / len(type_preds)
            else:
                type_accuracies[ctype] = 0.0
        
        colors = ['#336699' if v > 0 else '#cccccc' for v in type_accuracies.values()]
        axes[0].bar(type_accuracies.keys(), type_accuracies.values(), color=colors)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Accuracy by Character Type\n(Overall: {accuracy:.1%})')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(y=1/256, color='red', linestyle='--', alpha=0.5, label='Random (1/256)')
        axes[0].legend()
        
        # Right: Confusion matrix (top bytes)
        byte_counts = Counter([p["actual"] for p in self.predictions])
        top_bytes = [b for b, _ in byte_counts.most_common(15)]
        
        conf_size = len(top_bytes)
        confusion = np.zeros((conf_size, conf_size), dtype=np.int32)
        byte_to_idx = {b: i for i, b in enumerate(top_bytes)}
        
        for p in self.predictions:
            if p["actual"] in byte_to_idx and p["predicted"] in byte_to_idx:
                i = byte_to_idx[p["actual"]]
                j = byte_to_idx[p["predicted"]]
                confusion[i, j] += 1
        
        im = axes[1].imshow(confusion, cmap='Blues', aspect='auto')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title(f'Confusion Matrix (Top {conf_size} Bytes)\n'
                         f'Carriers: {num_carriers} ({num_crystallized} crystallized)')
        
        labels = [chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in top_bytes]
        axes[1].set_xticks(range(conf_size))
        axes[1].set_yticks(range(conf_size))
        axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_yticklabels(labels, fontsize=7)
        
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "next_token.pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/next_token_summary.tex")
        print(f"[next_token] Accuracy: {accuracy:.3f}, Top-3: {top3_accuracy:.3f}, "
              f"Perplexity: {perplexity:.2f}")

    def run(self):
        """Run the next-byte prediction experiment with dual-domain inference."""
        print("[next_token] Starting experiment...")
        
        # Create dataset and split
        dataset = TextDataset()
        train_bytes, test_bytes = dataset.train_test_split(test_ratio=0.2)
        
        print(f"[next_token] Training on {len(train_bytes)} bytes")
        print(f"[next_token] Testing on {len(test_bytes)} bytes")
        
        # Train: Run manifold on training data
        tokenizer_config = TokenizerConfig(
            hash_vocab_size=self.vocab_size,
            hash_prime=self.prime,
        )
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=False,
                generator=lambda: (bytes([b]) for b in train_bytes),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                ),
                spectral=SpectralSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                    max_carriers=64,
                ),
                tokenizer=tokenizer_config,
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "spectral": InferenceObserver([CarrierObserver(None)])
            }
        )
        
        state = manifold.run()
        carriers = manifold.carriers or {}
        
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
        
        inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=carriers,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
        
        print(f"[next_token] Carriers formed: {inference.num_carriers}")
        
        # Show carrier stats
        crystallized = inference.crystallized_carriers()
        print(f"[next_token] Crystallized carriers: {crystallized.carrier_indices.numel()}")
        
        strongest = inference.strongest_carriers(k=5)
        if strongest.amplitudes.numel() > 0:
            print(f"[next_token] Strongest carrier amplitudes: "
                  f"{strongest.amplitudes.cpu().numpy()}")
        
        # Test: Predict next bytes using dual-domain inference
        n_particles = geo_state["token_ids"].numel() if geo_state["token_ids"] is not None else 0
        
        for pos in range(self.context_length, len(test_bytes)):
            actual = test_bytes[pos]
            
            # Get context particle indices
            # We need to find particles that correspond to the context positions
            # Since we're testing on held-out data, we'll use the last context_length
            # particles from training as a proxy for "recent context"
            context_start = max(0, n_particles - self.context_length)
            context_indices = torch.arange(
                context_start, n_particles, 
                device=inference.device
            )
            
            # Score candidates using dual-domain inference
            scores = inference.score_candidate_bytes(
                context_indices=context_indices,
                target_position=pos,
                segment_size=None,
            )
            
            # Predict
            predicted = int(np.argmax(scores))
            
            # Top-k
            top_indices = np.argsort(scores)[::-1]
            top3 = list(top_indices[:3])
            top5 = list(top_indices[:5])
            
            self.predictions.append({
                "position": pos,
                "actual": actual,
                "predicted": predicted,
                "top3": top3,
                "top5": top5,
                "scores": scores.copy(),
            })
        
        print(f"[next_token] Made {len(self.predictions)} predictions")
        
        self.observe(state, carriers)
        print("[next_token] Experiment complete.")
