"""Kernel text "diffusion" (byte denoising) via Universal Tokenizer.

We corrupt a byte sequence by masking bytes, then reconstruct by sequentially
sampling bytes using carrier scores.

NON-CHEATING DESIGN:
====================
This experiment follows the masked language model paradigm but for bytes:
- Training: Learn from clean text (no masks during training)
- Inference: Mask random positions, reconstruct using learned patterns
- No access to ground truth during reconstruction

The key mechanism is "crystallization" - the carriers that form during training
encode patterns that can fill in missing information.

This is analogous to diffusion models but uses thermodynamic relaxation
instead of iterative denoising steps.

Writes:
- `paper/tables/text_diffusion_summary.tex`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any

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


class TextDiffusionPredictor:
    """Reconstruct masked bytes using dual-domain inference.
    
    Key insight: unmasked positions are "hot" particles that couple to carriers.
    Those carriers also couple to oscillators at masked positions.
    We can use the carrier structure to fill in the gaps.
    """
    
    def __init__(self, vocab_size: int = 4096, prime: int = 31):
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask_val = vocab_size - 1
        self.inference = None
    
    def learn_from_manifold(self, geo_state: Dict, spec_state: Dict):
        """Set up dual-domain inference from manifold state."""
        from sensorium.observers.dual_domain import DualDomainInference
        
        self.inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=spec_state,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
    
    def reconstruct(
        self,
        corrupted: bytes,
        mask_positions: List[int],
        context_window: int = 5,
    ) -> bytes:
        """Reconstruct masked positions using dual-domain inference.
        
        Strategy:
        1. Find "hot" particles (high energy) among unmasked positions
        2. Find carriers they couple to (switch to spectral)
        3. For each masked position, find byte that couples best to those carriers
        """
        result = bytearray(corrupted)
        remaining = set(mask_positions)
        
        if self.inference is None:
            return bytes(result)
        
        # Identify unmasked positions (context)
        all_positions = set(range(len(corrupted)))
        unmasked_positions = all_positions - set(mask_positions)
        
        # Iteratively fill in masked positions
        max_iters = len(mask_positions) + 10
        for iteration in range(max_iters):
            if not remaining:
                break
            
            best_pos = None
            best_byte = None
            best_score = -float('inf')
            
            for pos in remaining:
                # Find nearby unmasked positions for context
                context_positions = []
                for i in range(max(0, pos - context_window), min(len(result), pos + context_window + 1)):
                    if i not in remaining and i != pos:
                        context_positions.append(i)
                
                if not context_positions:
                    continue
                
                # Create context indices tensor
                context_indices = torch.tensor(
                    context_positions, 
                    device=self.inference.device,
                    dtype=torch.int64
                )
                
                # Score candidates using dual-domain inference
                scores = self.inference.score_candidate_bytes(
                    context_indices=context_indices,
                    target_position=pos,
                    segment_size=None,
                )
                
                predicted_byte = int(np.argmax(scores))
                score = scores[predicted_byte]
                
                if score > best_score:
                    best_score = score
                    best_pos = pos
                    best_byte = predicted_byte
            
            if best_pos is not None:
                result[best_pos] = best_byte
                remaining.remove(best_pos)
            else:
                # No progress, fill remaining with most common byte
                for pos in list(remaining):
                    result[pos] = ord(' ')  # Default to space
                    remaining.remove(pos)
        
        return bytes(result)


class KernelTextDiffusion(Experiment):
    """Text byte denoising/inpainting experiment."""
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        
        self.max_bytes = 1500
        self.mask_fracs = [0.1, 0.2, 0.3, 0.5]  # Test multiple corruption levels
        self.hash_vocab_size = 4096
        
        self.results: Dict[float, Dict[str, Any]] = {}

    def _get_sample_text(self) -> bytes:
        """Get sample text for training and testing."""
        # Try to read from paper/main.tex
        paper_path = self.paper_dir / "main.tex"
        if paper_path.exists():
            text = paper_path.read_bytes()[:self.max_bytes]
            if len(text) >= 100:
                return text
        
        # Fallback: use a sample text with patterns
        sample = """
The Sensorium Manifold is a thermodynamic computing substrate.
The system uses oscillators and carriers for pattern learning.
The physics engine processes particles in continuous space.
The spectral carriers couple distant oscillators via resonance.
The crystallization mechanism enables all-token prediction.
The holographic content addressable memory retrieves patterns.
The universal tokenizer maps bytes to oscillator frequencies.
""" * 10  # Repeat for more data
        
        return sample.strip().encode("utf-8")[:self.max_bytes]

    def observe(self, state: dict):
        """Generate paper artifacts."""
        if not self.results:
            print("Warning: No results collected")
            return
        
        import matplotlib.pyplot as plt
        
        # Summary table
        summary = {}
        for mask_frac, res in self.results.items():
            key = f"mask_{int(mask_frac*100)}pct"
            summary[f"{key}_char_accuracy"] = res["char_accuracy"]
            summary[f"{key}_exact_match"] = res["exact_match"]
            summary[f"{key}_levenshtein"] = res["levenshtein_dist"]
        
        self.write_kv_table("text_diffusion_summary", summary)
        
        # Figure: Accuracy vs corruption level
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        mask_fracs = sorted(self.results.keys())
        char_accs = [self.results[mf]["char_accuracy"] for mf in mask_fracs]
        exact_matches = [self.results[mf]["exact_match"] for mf in mask_fracs]
        
        # Left: Character accuracy
        axes[0].plot([m * 100 for m in mask_fracs], char_accs, 'o-', 
                    color='#336699', linewidth=2, markersize=10)
        axes[0].set_xlabel('Mask Percentage (%)', fontsize=12)
        axes[0].set_ylabel('Character Accuracy', fontsize=12)
        axes[0].set_title('Reconstruction Accuracy vs Corruption Level', fontsize=13)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3)
        
        # Add baseline reference
        axes[0].axhline(y=1/256, color='red', linestyle='--', 
                       alpha=0.5, label='Random baseline (1/256)')
        axes[0].legend()
        
        # Right: Example reconstruction
        ax = axes[1]
        ax.axis('off')
        
        # Show one example
        if mask_fracs:
            mf = mask_fracs[1] if len(mask_fracs) > 1 else mask_fracs[0]
            res = self.results[mf]
            
            original = res.get("original_sample", "")[:200]
            corrupted = res.get("corrupted_sample", "")[:200]
            reconstructed = res.get("reconstructed_sample", "")[:200]
            
            text = (
                f"Mask Rate: {mf*100:.0f}%\n\n"
                f"Original (first 200 chars):\n{original}\n\n"
                f"Corrupted:\n{corrupted}\n\n"
                f"Reconstructed:\n{reconstructed}"
            )
            
            ax.text(0.05, 0.95, text, fontsize=9, family='monospace',
                   verticalalignment='top', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "text_diffusion.pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/text_diffusion_summary.tex")

    def run(self):
        """Run text diffusion experiment at multiple corruption levels."""
        print("[text_diffusion] Starting experiment...")
        
        # Get sample text
        text_bytes = self._get_sample_text()
        print(f"[text_diffusion] Using {len(text_bytes)} bytes of text")
        
        # Split: train on first 80%, test reconstruction on last 20%
        split_idx = int(len(text_bytes) * 0.8)
        train_bytes = text_bytes[:split_idx]
        test_bytes = text_bytes[split_idx:]
        
        print(f"[text_diffusion] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
        
        # Train manifold on clean text
        tokenizer_config = TokenizerConfig(
            hash_vocab_size=4096,
            hash_prime=31,
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
        carriers = manifold.carriers or {}
        
        predictor = TextDiffusionPredictor(vocab_size=4096, prime=31)
        predictor.learn_from_manifold(geo_state, carriers)
        
        # Test reconstruction at different corruption levels
        rng = np.random.RandomState(42)
        
        for mask_frac in self.mask_fracs:
            print(f"[text_diffusion] Testing mask fraction: {mask_frac}")
            
            # Corrupt test text
            n_mask = int(len(test_bytes) * mask_frac)
            mask_positions = list(rng.choice(len(test_bytes), size=n_mask, replace=False))
            
            corrupted = bytearray(test_bytes)
            mask_byte = ord('?')  # Use '?' as mask token
            for pos in mask_positions:
                corrupted[pos] = mask_byte
            
            # Reconstruct
            reconstructed = predictor.reconstruct(
                bytes(corrupted), mask_positions, context_window=5
            )
            
            # Evaluate
            correct = 0
            for pos in mask_positions:
                if reconstructed[pos] == test_bytes[pos]:
                    correct += 1
            
            char_accuracy = correct / len(mask_positions) if mask_positions else 0.0
            exact_match = 1.0 if reconstructed == test_bytes else 0.0
            
            # Levenshtein distance (approximate)
            lev_dist = sum(1 for a, b in zip(reconstructed, test_bytes) if a != b)
            
            self.results[mask_frac] = {
                "char_accuracy": char_accuracy,
                "exact_match": exact_match,
                "levenshtein_dist": lev_dist,
                "n_masked": len(mask_positions),
                "original_sample": test_bytes.decode("utf-8", errors="replace"),
                "corrupted_sample": bytes(corrupted).decode("utf-8", errors="replace"),
                "reconstructed_sample": reconstructed.decode("utf-8", errors="replace"),
            }
            
            print(f"[text_diffusion] Mask {mask_frac*100:.0f}%: "
                  f"Char accuracy = {char_accuracy:.3f}, Lev dist = {lev_dist}")
        
        self.observe(state)
        print("[text_diffusion] Experiment complete.")
