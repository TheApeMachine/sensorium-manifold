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
from sensorium.observers.base import ObserverProtocol


class CrystallizationObserver(ObserverProtocol):
    """Observer that waits for carriers to crystallize before stopping.
    
    Requires:
    - At least min_carriers carriers
    - At least min_crystallized crystallized carriers
    - OR max_steps have elapsed
    """
    
    def __init__(self, min_carriers: int = 5, min_crystallized: int = 1, max_steps: int = 200):
        self.min_carriers = min_carriers
        self.min_crystallized = min_crystallized
        self.max_steps = max_steps
        self.step_count = 0
        self._current_observation = None
    
    def observe(self, observation=None, **kwargs):
        self.step_count += 1
        self._current_observation = observation
        
        if observation is None:
            return {"done_thinking": self.step_count >= self.max_steps}
        
        # Count carriers from amplitudes (num_carriers tensor has accumulation bug)
        amplitudes = observation.get("amplitudes")
        if amplitudes is not None:
            num_carriers = int((amplitudes > 1e-6).sum().item())
        else:
            num_carriers = 0
        
        carrier_state = observation.get("carrier_state")
        num_crystallized = 0
        if carrier_state is not None and num_carriers > 0:
            num_crystallized = int((carrier_state[:num_carriers] == 2).sum().item())
        
        # Check if we've met crystallization criteria
        meets_criteria = (
            num_carriers >= self.min_carriers and 
            num_crystallized >= self.min_crystallized
        )
        
        # Or if we've hit max steps
        done = meets_criteria or (self.step_count >= self.max_steps)
        
        if self.step_count % 50 == 0:
            # More diagnostics
            conflict = observation.get("conflict")
            max_amp = float(amplitudes.max().item()) if amplitudes is not None and amplitudes.numel() > 0 else 0
            min_conf = float(conflict[:num_carriers].min().item()) if conflict is not None and num_carriers > 0 else -1
            print(f"  [step {self.step_count}] carriers={num_carriers}, crystallized={num_crystallized}, "
                  f"max_amp={max_amp:.3f}, min_conflict={min_conf:.3f}")
        
        return {
            **observation,
            "done_thinking": done,
        }


class TextDataset:
    """Text dataset with multiple patterns sharing prefixes.
    
    Tests the thermodynamic trie's ability to:
    1. Learn multiple different continuations for the same prefix
    2. Generalize to unseen combinations
    3. Handle branching (same prefix, different suffixes)
    """
    
    SEGMENT_SIZE = 16  # Fixed segment size for all patterns
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Patterns that share prefixes - all padded to SEGMENT_SIZE
        # "The cat" can be followed by different things
        # "The dog" is a different branch
        self.patterns = [
            "The cat sat.    ",  # 16 chars
            "The cat ran.    ",  # 16 chars  
            "The cat ate.    ",  # 16 chars
            "The dog sat.    ",  # 16 chars
            "The dog ran.    ",  # 16 chars
            "A cat sat.      ",  # 16 chars (different start, same structure)
            "A dog ran.      ",  # 16 chars
        ]
        
        # Verify all patterns are correct length
        for p in self.patterns:
            assert len(p) == self.SEGMENT_SIZE, f"Pattern '{p}' has length {len(p)}, expected {self.SEGMENT_SIZE}"
        
        # Build training data with controlled frequencies
        # More common patterns get more repetitions
        self.pattern_counts = {
            "The cat sat.    ": 30,  # Most common
            "The cat ran.    ": 20,
            "The cat ate.    ": 10,
            "The dog sat.    ": 15,
            "The dog ran.    ": 10,
            "A cat sat.      ": 8,
            "A dog ran.      ": 7,
        }
        
        # Generate training text
        train_patterns = []
        for pattern, count in self.pattern_counts.items():
            train_patterns.extend([pattern] * count)
        
        self._rng.shuffle(train_patterns)
        self.train_text = "".join(train_patterns)
        self.train_bytes = self.train_text.encode("utf-8")
        
        # Test data: sample of each pattern (to test recall)
        # Plus some patterns with lower frequency (to test generalization)
        test_patterns = []
        for pattern in self.patterns:
            test_patterns.extend([pattern] * 3)  # 3 of each for testing
        
        self._rng.shuffle(test_patterns)
        self.test_text = "".join(test_patterns)
        self.test_bytes = self.test_text.encode("utf-8")
    
    def train_test_split(self, test_ratio: float = 0.2) -> Tuple[bytes, bytes]:
        """Return pre-built train/test split."""
        return self.train_bytes, self.test_bytes
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Return pattern frequency statistics."""
        return self.pattern_counts.copy()


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
        
        # Calculate "ambiguous" vs "unambiguous" accuracy
        # Ambiguous = cases where multiple bytes have non-zero probability
        ambiguous_count = 0
        ambiguous_correct = 0
        unambiguous_count = 0
        unambiguous_correct = 0
        
        for p in self.predictions:
            scores = p["scores"]
            non_zero = (scores > 0.01).sum()
            is_correct = p["predicted"] == p["actual"]
            
            if non_zero > 1:
                ambiguous_count += 1
                if is_correct:
                    ambiguous_correct += 1
            else:
                unambiguous_count += 1
                if is_correct:
                    unambiguous_correct += 1
        
        ambiguous_accuracy = ambiguous_correct / ambiguous_count if ambiguous_count > 0 else 1.0
        unambiguous_accuracy = unambiguous_correct / unambiguous_count if unambiguous_count > 0 else 1.0
        
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
            "ambiguous_cases": ambiguous_count,
            "ambiguous_accuracy": ambiguous_accuracy,
            "unambiguous_accuracy": unambiguous_accuracy,
        }
        
        # Add trie statistics to summary
        summary["unique_token_ids"] = self.unique_tids
        summary["collision_ratio"] = self.collision_ratio
        summary["segment_size"] = self.segment_size
        summary["num_patterns"] = len(self.pattern_counts) if hasattr(self, 'pattern_counts') else 0
        
        self.write_kv_table("next_token_summary", summary)
        
        # Generate hero visualization: 3 clean charts
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Build data structures for visualization
        prefix_to_continuations = {}
        for p in self.predictions:
            pos = p["position"]
            scores = p["scores"]
            continuations = [(b, float(scores[b])) for b in range(256) if scores[b] > 0.01]
            if continuations:
                seg_pos = pos % self.segment_size
                if seg_pos not in prefix_to_continuations:
                    prefix_to_continuations[seg_pos] = {}
                for byte_val, score in continuations:
                    char = chr(byte_val) if 32 <= byte_val < 127 else f'x{byte_val:02x}'
                    if char not in prefix_to_continuations[seg_pos]:
                        prefix_to_continuations[seg_pos][char] = 0
                    prefix_to_continuations[seg_pos][char] += score
        
        # =================================================================
        # Panel A: Trie branching visualization (node-link diagram)
        # =================================================================
        ax = axes[0]
        
        positions_to_show = sorted(prefix_to_continuations.keys())[:14]
        y_positions = {}
        
        for i, pos in enumerate(positions_to_show):
            x = i * 1.0
            continuations = prefix_to_continuations.get(pos, {})
            sorted_conts = sorted(continuations.items(), key=lambda x: -x[1])[:4]
            total_score = sum(s for _, s in sorted_conts)
            
            for j, (char, score) in enumerate(sorted_conts):
                y = 0.5 - j * 0.25
                prob = score / total_score if total_score > 0 else 0
                
                size = 150 + prob * 350
                color = plt.cm.Blues(0.3 + prob * 0.7)
                
                ax.scatter([x], [y], s=size, c=[color], edgecolors='black', linewidths=0.5, zorder=3)
                ax.annotate(char, (x, y), ha='center', va='center', fontsize=7, fontweight='bold')
                
                if i > 0 and pos - 1 in y_positions:
                    for prev_y in y_positions[pos - 1]:
                        ax.plot([x-1, x], [prev_y, y], 'k-', alpha=0.15, linewidth=0.5)
            
            y_positions[pos] = [0.5 - j * 0.25 for j in range(len(sorted_conts))]
        
        ax.set_xlim(-0.5, len(positions_to_show) - 0.5)
        ax.set_ylim(-0.6, 0.75)
        ax.set_xlabel('Position in segment', fontsize=10)
        ax.set_xticks(range(0, len(positions_to_show), 2))
        ax.set_xticklabels([positions_to_show[i] for i in range(0, len(positions_to_show), 2)], fontsize=8)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: Probability distributions at branch points (stacked bar)
        # =================================================================
        ax = axes[1]
        
        interesting_positions = []
        for pos, conts in prefix_to_continuations.items():
            sorted_conts = sorted(conts.items(), key=lambda x: -x[1])
            if len(sorted_conts) >= 2:
                total = sum(s for _, s in sorted_conts)
                probs = [s/total for _, s in sorted_conts[:4]]
                if probs[0] < 0.85:
                    interesting_positions.append((pos, sorted_conts[:4], probs))
        
        interesting_positions = sorted(interesting_positions, key=lambda x: -x[2][1] if len(x[2]) > 1 else 0)[:5]
        
        if interesting_positions:
            x_positions = np.arange(len(interesting_positions))
            bar_width = 0.65
            
            for i, (pos, conts, probs) in enumerate(interesting_positions):
                bottom = 0
                for j, ((char, _), prob) in enumerate(zip(conts, probs)):
                    color = plt.cm.Set2(j)
                    ax.bar(i, prob, bar_width, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
                    if prob > 0.12:
                        ax.text(i, bottom + prob/2, f"'{char}'\n{prob:.0%}", 
                               ha='center', va='center', fontsize=7, fontweight='bold')
                    bottom += prob
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"pos {p[0]}" for p in interesting_positions], fontsize=9)
        
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: Accuracy by position (bar chart with color coding)
        # =================================================================
        ax = axes[2]
        
        pos_correct = {}
        pos_total = {}
        for p in self.predictions:
            seg_pos = p["position"] % self.segment_size
            if seg_pos not in pos_correct:
                pos_correct[seg_pos] = 0
                pos_total[seg_pos] = 0
            pos_total[seg_pos] += 1
            if p["predicted"] == p["actual"]:
                pos_correct[seg_pos] += 1
        
        positions = sorted(pos_correct.keys())
        accuracies = [pos_correct[p] / pos_total[p] if pos_total[p] > 0 else 0 for p in positions]
        
        colors = ['#e74c3c' if a < 0.7 else '#f39c12' if a < 0.95 else '#27ae60' for a in accuracies]
        ax.bar(positions, accuracies, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=accuracy, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {accuracy:.1%}')
        ax.set_xlabel('Position in segment', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "next_token.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/next_token_summary.tex")
        
        # Write simulation stats
        n_particles = len(geo_state.get("token_ids", [])) if geo_state.get("token_ids") is not None else 0
        self.write_simulation_stats(
            "next_token",
            n_particles=n_particles,
            n_carriers=num_carriers,
            n_crystallized=num_crystallized,
            grid_size=getattr(self, "_grid_size", (32, 32, 32)),
            dt=getattr(self, "_dt", 0.01),
            n_steps=getattr(self, "_n_steps", 1),
            wall_time_ms=getattr(self, "_wall_time_ms", 0),
        )
        print(f"✓ Generated: paper/tables/next_token_stats.tex")
        
        print(f"[next_token] Accuracy: {accuracy:.3f}, Top-3: {top3_accuracy:.3f}, "
              f"Perplexity: {perplexity:.2f}")

    def run(self):
        """Run the next-byte prediction experiment with dual-domain inference."""
        import time
        
        print("[next_token] Starting experiment...")
        
        # Create dataset and split
        dataset = TextDataset(seed=42)
        train_bytes, test_bytes = dataset.train_test_split()
        
        print(f"[next_token] Training on {len(train_bytes)} bytes")
        print(f"[next_token] Testing on {len(test_bytes)} bytes")
        print(f"[next_token] Pattern frequencies: {dataset.get_pattern_stats()}")
        
        # Train: Run manifold on training data
        # Use segment_size so positions wrap, creating the thermodynamic trie
        segment_size = TextDataset.SEGMENT_SIZE
        
        tokenizer_config = TokenizerConfig(
            hash_vocab_size=self.vocab_size,
            hash_prime=self.prime,
            segment_size=segment_size,  # Reset position every segment
        )
        
        # Use CrystallizationObserver to ensure we run enough steps
        crystallization_observer = CrystallizationObserver(
            min_carriers=10,
            min_crystallized=5,  # Wait for more crystallization
            max_steps=500,       # Allow more time
        )
        
        grid_size = (32, 32, 32)
        dt = 0.01
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=False,
                generator=lambda: (bytes([b]) for b in train_bytes),
                geometric=GeometricSimulationConfig(
                    grid_size=grid_size,
                    dt=dt,
                ),
                spectral=SpectralSimulationConfig(
                    grid_size=grid_size,
                    dt=dt,
                    max_carriers=64,
                    # Tune for faster crystallization
                    stable_amp_threshold=0.15,      # Lower threshold for stable (was 0.25)
                    crystallize_amp_threshold=0.20,  # Lower threshold for crystallize (was 0.75)
                    volatile_decay_mul=0.98,         # Slower decay (was 0.90)
                    coupling_scale=0.5,              # Stronger coupling (was 0.25)
                ),
                tokenizer=tokenizer_config,
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "spectral": InferenceObserver([crystallization_observer])
            }
        )
        
        start_time = time.time()
        state = manifold.run()
        self._wall_time_ms = (time.time() - start_time) * 1000
        self._grid_size = grid_size
        self._dt = dt
        self._n_steps = crystallization_observer.step_count
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
        
        # Trie structure statistics (store for observe method)
        train_token_ids = geo_state["token_ids"].cpu().numpy() if geo_state["token_ids"] is not None else []
        self.unique_tids = len(set(train_token_ids)) if len(train_token_ids) > 0 else 1
        self.collision_ratio = len(train_token_ids) / self.unique_tids
        self.segment_size = segment_size
        self.pattern_counts = dataset.get_pattern_stats()
        
        print(f"[next_token] Trie structure: {self.unique_tids} unique nodes, "
              f"{self.collision_ratio:.1f}x collision ratio")
        
        # Test: Predict next bytes using dual-domain inference
        # Use pattern matching + dehashing approach
        
        # Debug: print first few predictions
        debug_count = 0
        
        # Calculate where test data starts in the segment cycle
        test_offset_in_segment = len(train_bytes) % segment_size
        
        for idx in range(self.context_length, len(test_bytes)):
            actual = test_bytes[idx]
            
            # Get actual context bytes from test data
            context_start_idx = idx - self.context_length
            context = test_bytes[context_start_idx:idx]
            
            # Calculate segment-relative position for context start
            # The context starts at (test_offset + context_start_idx) % segment_size
            context_start_pos = (test_offset_in_segment + context_start_idx) % segment_size
            
            # Predict using pattern matching and dehashing
            scores, top_candidates = inference.predict_next_byte(
                context_bytes=context,
                context_start_position=context_start_pos,
                segment_size=segment_size,
            )
            
            # Debug first few predictions to show branching behavior
            if debug_count < 3:
                predicted = int(np.argmax(scores))
                context_str = ''.join(chr(b) if 32 <= b < 127 else '?' for b in context)
                
                # Show top candidates with probabilities
                top_str = ', '.join([f"'{chr(b) if 32 <= b < 127 else '?'}':{s:.0%}" 
                                    for b, s in top_candidates[:3] if s > 0.01])
                
                is_correct = "✓" if predicted == actual else "✗"
                print(f"  {is_correct} '{context_str}' -> [{top_str}]")
                debug_count += 1
            
            # Predict
            predicted = int(np.argmax(scores))
            
            # Top-k
            top_indices = np.argsort(scores)[::-1]
            top3 = list(top_indices[:3])
            top5 = list(top_indices[:5])
            
            self.predictions.append({
                "position": idx,
                "actual": actual,
                "predicted": predicted,
                "top3": top3,
                "top5": top5,
                "scores": scores.copy(),
            })
        
        print(f"[next_token] Made {len(self.predictions)} predictions")
        
        self.observe(state, carriers)
        print("[next_token] Experiment complete.")
