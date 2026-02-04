"""Kernel-based rule-shift experiment.

Demonstrates online adaptation to distributional shifts.

EXPERIMENT DESIGN:
==================
Phase 1 (Forward): The sequence "The cat sat on the mat." repeats
Phase 2 (Reverse): The sequence reverses to "mat the on sat cat The."

The system must:
1. Learn the forward transition patterns during Phase 1
2. Rapidly adapt when the rule shifts
3. Learn reverse transitions online during Phase 2

KEY: Uses the thermodynamic trie mechanism - the same (byte, position) pairs
create hash collisions, enabling pattern learning.

Produces:
- `paper/tables/rule_shift_summary.tex`
- `paper/figures/rule_shift.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig


class KernelRuleShift(Experiment):
    """Rule-shift adaptation experiment using thermodynamic trie."""
    
    SEGMENT_SIZE = 24  # Fixed segment size for trie formation
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        # Tokenizer params
        self.vocab_size = 4096
        self.prime = 31
        self.mask = self.vocab_size - 1
        self.inv_prime = pow(self.prime, -1, self.vocab_size)
        
        # Phrases (padded to SEGMENT_SIZE)
        self.forward_phrase = "The cat sat on the mat."
        self.reverse_phrase = "mat the on sat cat The."
        
        # Pad to segment size
        self.forward_phrase = self.forward_phrase.ljust(self.SEGMENT_SIZE)[:self.SEGMENT_SIZE]
        self.reverse_phrase = self.reverse_phrase.ljust(self.SEGMENT_SIZE)[:self.SEGMENT_SIZE]
        
        # Experiment params
        self.forward_reps = 50  # Repetitions of forward phrase
        self.reverse_reps = 50  # Repetitions of reverse phrase
        self.context_length = 8
        self.eval_every = 5  # Evaluate every N repetitions
        
        # Results
        self.accuracy_history: List[Dict[str, Any]] = []
    
    def run(self):
        import time
        
        print(f"[rule_shift] Starting experiment...")
        print(f"[rule_shift] Forward phrase: '{self.forward_phrase}'")
        print(f"[rule_shift] Reverse phrase: '{self.reverse_phrase}'")
        print(f"[rule_shift] Segment size: {self.SEGMENT_SIZE}")
        
        # Build training data: forward phase then reverse phase
        forward_text = self.forward_phrase * self.forward_reps
        reverse_text = self.reverse_phrase * self.reverse_reps
        full_text = forward_text + reverse_text
        train_bytes = full_text.encode("utf-8")
        
        print(f"[rule_shift] Total training: {len(train_bytes)} bytes")
        print(f"[rule_shift] Phase switch at byte {len(forward_text)}")
        
        # Generator for manifold
        def train_generator():
            yield train_bytes
        
        # Configure manifold
        grid_size = (64, 64, 64)
        dt = 0.01
        
        cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=self.vocab_size,
                hash_prime=self.prime,
                segment_size=self.SEGMENT_SIZE,
            ),
            spectral=SpectralSimulationConfig(
                max_carriers=64,
                stable_amp_threshold=0.15,
                crystallize_amp_threshold=0.20,
                volatile_decay_mul=0.98,
                coupling_scale=0.5,
                grid_size=grid_size,
                dt=dt,
            ),
            generator=train_generator,
        )
        
        manifold = Manifold(cfg)
        
        start_time = time.time()
        state = manifold.run()
        wall_time_ms = (time.time() - start_time) * 1000
        
        # Get tokenized data
        token_ids = state.get("token_ids")
        if token_ids is None:
            print("[rule_shift] ERROR: No token_ids in state")
            return
        
        token_ids_np = token_ids.cpu().numpy()
        energies_np = state.get("energies", torch.ones(len(token_ids))).cpu().numpy()
        n_particles = len(token_ids_np)
        
        print(f"[rule_shift] Tokenized {n_particles} particles")
        
        # Evaluate accuracy at different points during training
        # KEY: We only use data BEFORE the current position for prediction
        # This simulates online learning where the system adapts incrementally
        
        phase_switch_byte = len(forward_text)
        
        for rep in range(self.eval_every, self.forward_reps + self.reverse_reps, self.eval_every):
            # Current position in stream (end of data seen so far)
            current_byte = rep * self.SEGMENT_SIZE
            
            # Determine phase
            phase = "forward" if current_byte < phase_switch_byte else "reverse"
            
            # Test phrase to predict (current segment)
            if phase == "forward":
                test_phrase = self.forward_phrase
            else:
                test_phrase = self.reverse_phrase
            
            correct = 0
            total = 0
            
            # For each position in the test phrase, try to predict using PAST data only
            for seg_pos in range(self.context_length, self.SEGMENT_SIZE):
                # Build context from the test phrase
                context_tids = []
                for i in range(self.context_length):
                    ctx_pos = seg_pos - self.context_length + i
                    byte_val = ord(test_phrase[ctx_pos])
                    tid = (byte_val * self.prime + ctx_pos) & self.mask
                    context_tids.append(tid)
                
                # Predict using only data seen BEFORE this rep
                # This is the key: we limit the trie to past data
                data_limit = current_byte
                
                predictions = self._predict_next_byte(
                    token_ids_np[:data_limit],
                    energies_np[:data_limit],
                    context_tids,
                    seg_pos,
                )
                
                # Get actual byte from test phrase
                actual_byte = ord(test_phrase[seg_pos])
                predicted_byte = predictions[0][0] if predictions else 128
                
                if predicted_byte == actual_byte:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            
            self.accuracy_history.append({
                "rep": rep,
                "byte_position": current_byte,
                "phase": phase,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            })
            
            print(f"[rule_shift] Rep {rep}: {phase} accuracy = {accuracy:.3f} ({correct}/{total})")
        
        # Count carriers
        carriers = manifold.carriers or {}
        amplitudes = carriers.get("amplitudes")
        n_carriers = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
        crystallized = carriers.get("crystallized")
        n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
        
        # Generate artifacts
        self._generate_table()
        self._generate_figure()
        
        # Write simulation stats
        self.write_simulation_stats(
            "rule_shift",
            n_particles=n_particles,
            n_carriers=n_carriers,
            n_crystallized=n_crystallized,
            grid_size=grid_size,
            dt=dt,
            n_steps=1,  # Single pass tokenization
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/rule_shift_stats.tex")
        
        print(f"[rule_shift] Experiment complete.")
    
    def _predict_next_byte(
        self,
        token_ids: np.ndarray,
        energies: np.ndarray,
        context_tids: List[int],
        target_seg_pos: int,
    ) -> List[Tuple[int, float]]:
        """Predict next byte using trie pattern matching."""
        
        scores = np.zeros(256, dtype=np.float32)
        n_particles = len(token_ids)
        context_len = len(context_tids)
        
        # Find sequences that match the context pattern
        for start_idx in range(n_particles - context_len):
            # Check if tokens match context
            match = True
            for j, ctx_tid in enumerate(context_tids):
                if token_ids[start_idx + j] != ctx_tid:
                    match = False
                    break
            
            if match:
                # Get the "next" particle
                next_idx = start_idx + context_len
                if next_idx < n_particles:
                    next_tid = int(token_ids[next_idx])
                    next_energy = float(energies[next_idx])
                    
                    # Dehash to get byte
                    byte_val = self._dehash(next_tid, target_seg_pos)
                    if 0 <= byte_val < 256:
                        scores[byte_val] += next_energy
        
        # Normalize and get top predictions
        if scores.sum() > 0:
            scores = scores / scores.sum()
        
        top_indices = np.argsort(scores)[::-1][:5]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _dehash(self, token_id: int, position: int) -> int:
        """Reverse the hash to get the original byte value."""
        target = (token_id - position) & self.mask
        byte_val = (target * self.inv_prime) & self.mask
        return byte_val
    
    def _generate_table(self):
        """Generate LaTeX table with results."""
        
        # Compute summary metrics
        forward_accs = [r["accuracy"] for r in self.accuracy_history if r["phase"] == "forward"]
        reverse_accs = [r["accuracy"] for r in self.accuracy_history if r["phase"] == "reverse"]
        
        forward_baseline = np.mean(forward_accs) if forward_accs else 0.0
        forward_final = forward_accs[-1] if forward_accs else 0.0
        reverse_initial = reverse_accs[0] if reverse_accs else 0.0
        reverse_final = reverse_accs[-1] if reverse_accs else 0.0
        
        # Find recovery point (when reverse accuracy reaches 80% of forward baseline)
        recovery_rep = None
        threshold = forward_baseline * 0.8
        for r in self.accuracy_history:
            if r["phase"] == "reverse" and r["accuracy"] >= threshold:
                recovery_rep = r["rep"] - self.forward_reps
                break
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Rule-shift adaptation results. The manifold learns forward transitions, then adapts online when the sequence reverses. Recovery time measures how quickly the system regains baseline accuracy after the rule shift.}
\label{tab:rule_shift}
\begin{tabular}{l c c}
\toprule
\textbf{Metric} & \textbf{Forward Phase} & \textbf{Reverse Phase} \\
\midrule
Mean accuracy & """ + f"{forward_baseline*100:.1f}\\%" + r""" & """ + f"{(np.mean(reverse_accs) if reverse_accs else 0)*100:.1f}\\%" + r""" \\
Final accuracy & """ + f"{forward_final*100:.1f}\\%" + r""" & """ + f"{reverse_final*100:.1f}\\%" + r""" \\
Initial accuracy & --- & """ + f"{reverse_initial*100:.1f}\\%" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Adaptation Dynamics}} \\
\quad Phase switch (rep) & \multicolumn{2}{c}{""" + str(self.forward_reps) + r"""} \\
\quad Recovery (reps after switch) & \multicolumn{2}{c}{""" + (str(recovery_rep) if recovery_rep else "N/A") + r"""} \\
\quad Segment size & \multicolumn{2}{c}{""" + str(self.SEGMENT_SIZE) + r"""} \\
\quad Context length & \multicolumn{2}{c}{""" + str(self.context_length) + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "rule_shift_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def _generate_figure(self):
        """Generate 3-panel visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Extract data
        reps = [r["rep"] for r in self.accuracy_history]
        accuracies = [r["accuracy"] for r in self.accuracy_history]
        phases = [r["phase"] for r in self.accuracy_history]
        
        forward_reps = [r for r, p in zip(reps, phases) if p == "forward"]
        forward_accs = [a for a, p in zip(accuracies, phases) if p == "forward"]
        reverse_reps = [r for r, p in zip(reps, phases) if p == "reverse"]
        reverse_accs = [a for a, p in zip(accuracies, phases) if p == "reverse"]
        
        # =================================================================
        # Panel A: Accuracy over time with phase transition
        # =================================================================
        ax = axes[0]
        
        ax.plot(forward_reps, forward_accs, 'o-', color='#336699', 
               linewidth=2, markersize=6, label='Forward')
        ax.plot(reverse_reps, reverse_accs, 's-', color='#4C994C',
               linewidth=2, markersize=6, label='Reverse')
        
        ax.axvline(x=self.forward_reps, color='red', linestyle='--', 
                  linewidth=2, label='Rule Shift')
        
        if forward_accs:
            ax.axhline(y=np.mean(forward_accs), color='#336699', linestyle=':', 
                      alpha=0.5)
        
        ax.set_xlabel("Training repetition", fontsize=10)
        ax.set_ylabel("Next-byte accuracy", fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: Accuracy comparison bar chart
        # =================================================================
        ax = axes[1]
        
        x_pos = np.arange(4)
        values = [
            np.mean(forward_accs) if forward_accs else 0,
            forward_accs[-1] if forward_accs else 0,
            reverse_accs[0] if reverse_accs else 0,
            reverse_accs[-1] if reverse_accs else 0,
        ]
        labels = ["Fwd Mean", "Fwd Final", "Rev Initial", "Rev Final"]
        colors = ['#336699', '#336699', '#4C994C', '#4C994C']
        
        bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.0%}", ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: Adaptation rate (accuracy delta from initial)
        # =================================================================
        ax = axes[2]
        
        if reverse_accs and len(reverse_accs) > 1:
            initial = reverse_accs[0]
            deltas = [acc - initial for acc in reverse_accs]
            delta_reps = [r - self.forward_reps for r in reverse_reps]
            
            ax.fill_between(delta_reps, 0, deltas, alpha=0.3, color='#4C994C')
            ax.plot(delta_reps, deltas, 'o-', color='#4C994C', linewidth=2, markersize=6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            ax.set_xlabel("Reps after rule shift", fontsize=10)
            ax.set_ylabel("Accuracy gain from initial", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "rule_shift.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
