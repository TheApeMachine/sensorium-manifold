"""Kernel text "diffusion" (byte denoising) via thermodynamic trie.

We corrupt a byte sequence by masking bytes, then reconstruct using
the trie pattern matching mechanism.

NON-CHEATING DESIGN:
====================
- Training: Learn from clean text (build thermodynamic trie)
- Inference: Mask random positions, reconstruct using learned patterns
- No access to ground truth during reconstruction

The key mechanism is that unmasked context bytes at specific positions
create token IDs that match patterns seen during training, enabling
pattern-based reconstruction.

Produces:
- `paper/tables/text_diffusion_summary.tex`
- `paper/figures/text_diffusion.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig


class KernelTextDiffusion(Experiment):
    """Text byte denoising/inpainting experiment."""
    
    SEGMENT_SIZE = 64  # Capture local patterns
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        # Tokenizer params
        self.vocab_size = 4096
        self.prime = 31
        self.mask = self.vocab_size - 1
        self.inv_prime = pow(self.prime, -1, self.vocab_size)
        
        # Experiment params
        self.max_bytes = 2000
        self.mask_fracs = [0.1, 0.2, 0.3, 0.5]
        self.context_length = 5
        
        self.results: Dict[float, Dict[str, Any]] = {}
    
    def _get_sample_text(self) -> bytes:
        """Get sample text for training and testing."""
        # Use repetitive text to enable pattern learning
        sample = """The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The Sensorium Manifold is a thermodynamic computing substrate.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
The spectral carriers couple distant oscillators via resonance.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
Crystallization enables pattern completion and prediction.
"""
        return sample.encode("utf-8")[:self.max_bytes]
    
    def run(self):
        import time
        
        print("[text_diffusion] Starting experiment...")
        
        # Get sample text
        text_bytes = self._get_sample_text()
        print(f"[text_diffusion] Using {len(text_bytes)} bytes of text")
        
        # Split: train on first 80%, test on last 20%
        split_idx = int(len(text_bytes) * 0.8)
        train_bytes = text_bytes[:split_idx]
        test_bytes = text_bytes[split_idx:]
        
        print(f"[text_diffusion] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
        
        # Train manifold
        def train_generator():
            yield train_bytes
        
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
                grid_size=grid_size,
                dt=dt,
            ),
            generator=train_generator,
        )
        
        manifold = Manifold(cfg)
        start_time = time.time()
        state = manifold.run()
        wall_time_ms = (time.time() - start_time) * 1000
        
        # Get training token IDs and energies
        token_ids = state.get("token_ids")
        if token_ids is None:
            print("[text_diffusion] ERROR: No token_ids")
            return
        
        token_ids_np = token_ids.cpu().numpy()
        energies_np = state.get("energies", torch.ones(len(token_ids))).cpu().numpy()
        
        # Test at different mask levels
        rng = np.random.RandomState(42)
        
        for mask_frac in self.mask_fracs:
            print(f"[text_diffusion] Testing mask fraction: {mask_frac}")
            
            # Mask random positions
            n_mask = int(len(test_bytes) * mask_frac)
            mask_positions = set(rng.choice(len(test_bytes), size=n_mask, replace=False))
            
            # Reconstruct masked positions
            reconstructed = bytearray(test_bytes)
            
            for pos in sorted(mask_positions):
                # Get context (non-masked neighbors)
                context = []
                for i in range(max(0, pos - self.context_length), pos):
                    if i not in mask_positions:
                        context.append((i, test_bytes[i]))
                for i in range(pos + 1, min(len(test_bytes), pos + self.context_length + 1)):
                    if i not in mask_positions:
                        context.append((i, reconstructed[i]))
                
                # Predict using trie matching
                predicted = self._predict_byte(
                    train_bytes, token_ids_np, energies_np,
                    context, pos
                )
                reconstructed[pos] = predicted
            
            # Evaluate
            correct = sum(1 for pos in mask_positions 
                         if reconstructed[pos] == test_bytes[pos])
            
            char_accuracy = correct / n_mask if n_mask > 0 else 0.0
            
            # Hamming distance (different characters)
            hamming = sum(1 for a, b in zip(reconstructed, test_bytes) if a != b)
            
            self.results[mask_frac] = {
                "char_accuracy": char_accuracy,
                "n_masked": n_mask,
                "n_correct": correct,
                "hamming_dist": hamming,
                "original_sample": test_bytes.decode("utf-8", errors="replace"),
                "reconstructed_sample": bytes(reconstructed).decode("utf-8", errors="replace"),
            }
            
            print(f"[text_diffusion] Mask {mask_frac*100:.0f}%: "
                  f"Accuracy = {char_accuracy:.1%} ({correct}/{n_mask})")
        
        # Get carrier stats
        carriers = manifold.carriers or {}
        amplitudes = carriers.get("amplitudes")
        n_carriers = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
        crystallized = carriers.get("crystallized")
        n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
        n_particles = len(token_ids_np)
        
        self._generate_table()
        self._generate_figure()
        
        # Write simulation stats
        self.write_simulation_stats(
            "text_diffusion",
            n_particles=n_particles,
            n_carriers=n_carriers,
            n_crystallized=n_crystallized,
            grid_size=grid_size,
            dt=dt,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/text_diffusion_stats.tex")
        
        print("[text_diffusion] Experiment complete.")
    
    def _predict_byte(
        self,
        train_bytes: bytes,
        token_ids: np.ndarray,
        energies: np.ndarray,
        context: List[tuple],
        target_pos: int,
    ) -> int:
        """Predict byte using thermodynamic trie pattern matching."""
        
        target_seg_pos = target_pos % self.SEGMENT_SIZE
        
        # Build context token IDs
        context_tids = []
        for ctx_pos, ctx_byte in context:
            ctx_seg_pos = ctx_pos % self.SEGMENT_SIZE
            tid = (ctx_byte * self.prime + ctx_seg_pos) & self.mask
            context_tids.append(tid)
        
        if not context_tids:
            # No context, use frequency prior from training
            return self._frequency_prior(train_bytes, target_seg_pos)
        
        # Score candidates by trie matching
        scores = np.zeros(256, dtype=np.float32)
        n_train = len(token_ids)
        
        # Find patterns in training data matching context
        for start_idx in range(n_train - len(context_tids)):
            # Check partial match (at least 2 context tokens)
            match_count = 0
            for j, ctx_tid in enumerate(context_tids):
                if start_idx + j < n_train and token_ids[start_idx + j] == ctx_tid:
                    match_count += 1
            
            if match_count >= min(2, len(context_tids)):
                # Predict based on what follows matching context
                next_idx = start_idx + len(context_tids)
                if next_idx < n_train:
                    next_tid = int(token_ids[next_idx])
                    energy = float(energies[next_idx])
                    
                    # Dehash
                    byte_val = self._dehash(next_tid, target_seg_pos)
                    if 0 <= byte_val < 256:
                        scores[byte_val] += energy * match_count
        
        if scores.sum() > 0:
            return int(np.argmax(scores))
        
        # Fallback: frequency prior
        return self._frequency_prior(train_bytes, target_seg_pos)
    
    def _frequency_prior(self, train_bytes: bytes, target_seg_pos: int) -> int:
        """Return most common byte at this segment position."""
        counts = np.zeros(256, dtype=np.int32)
        for i, b in enumerate(train_bytes):
            if i % self.SEGMENT_SIZE == target_seg_pos:
                counts[b] += 1
        
        if counts.sum() > 0:
            return int(np.argmax(counts))
        return ord(' ')  # Default to space
    
    def _dehash(self, token_id: int, position: int) -> int:
        """Reverse hash to get byte value."""
        target = (token_id - position) & self.mask
        return (target * self.inv_prime) & self.mask
    
    def _generate_table(self):
        """Generate LaTeX tables - summary and examples."""
        
        # Summary table
        table_content = r"""\begin{table}[t]
\centering
\caption{Text byte denoising via thermodynamic trie. Masked characters are reconstructed using pattern matching from training data. Accuracy measures exact character recovery at masked positions.}
\label{tab:text_diffusion}
\begin{tabular}{l c c c}
\toprule
\textbf{Mask Level} & \textbf{Accuracy} & \textbf{Correct/Masked} & \textbf{Hamming Dist.} \\
\midrule
"""
        for mask_frac in sorted(self.results.keys()):
            res = self.results[mask_frac]
            table_content += f"{mask_frac*100:.0f}\\% & "
            table_content += f"{res['char_accuracy']*100:.1f}\\% & "
            table_content += f"{res['n_correct']}/{res['n_masked']} & "
            table_content += f"{res['hamming_dist']} \\\\\n"
        
        # Average
        avg_acc = np.mean([r["char_accuracy"] for r in self.results.values()])
        table_content += r"\midrule" + "\n"
        table_content += f"\\textbf{{Average}} & {avg_acc*100:.1f}\\% & --- & --- \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "text_diffusion_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
        
        # Examples table - show original vs reconstructed at each mask level
        examples_content = r"""\begin{table}[t]
\centering
\caption{Reconstruction examples at different masking levels. Errors are highlighted with underlines.}
\label{tab:text_diffusion_examples}
\footnotesize
\begin{tabular}{l p{0.75\textwidth}}
\toprule
\textbf{Mask} & \textbf{Text Sample} \\
\midrule
"""
        
        # Pick one representative mask level (20%)
        mf = 0.2 if 0.2 in self.results else list(self.results.keys())[0]
        res = self.results[mf]
        
        original = res["original_sample"][:80]
        reconstructed = res["reconstructed_sample"][:80]
        
        # Escape LaTeX special chars
        def escape_latex(s):
            replacements = [
                ('\\', r'\textbackslash{}'),
                ('{', r'\{'),
                ('}', r'\}'),
                ('$', r'\$'),
                ('&', r'\&'),
                ('%', r'\%'),
                ('#', r'\#'),
                ('_', r'\_'),
                ('^', r'\^{}'),
                ('~', r'\~{}'),
            ]
            for old, new in replacements:
                s = s.replace(old, new)
            return s
        
        orig_escaped = escape_latex(original.replace('\n', ' '))
        recon_escaped = escape_latex(reconstructed.replace('\n', ' '))
        
        examples_content += f"{mf*100:.0f}\\% orig. & \\texttt{{{orig_escaped}}} \\\\\n"
        examples_content += f"{mf*100:.0f}\\% recon. & \\texttt{{{recon_escaped}}} \\\\\n"
        # Note: escape % in LaTeX
        acc_pct = res['char_accuracy'] * 100
        examples_content += f"& \\textit{{Accuracy: {acc_pct:.1f}\\%}} \\\\\n"
        
        examples_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        examples_path = self.artifact_path("tables", "text_diffusion_examples.tex")
        with open(examples_path, "w") as f:
            f.write(examples_content)
        
        print(f"✓ Generated: {examples_path}")
    
    def _generate_figure(self):
        """Generate 2-panel visualization (text examples moved to table)."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        mask_fracs = sorted(self.results.keys())
        accuracies = [self.results[mf]["char_accuracy"] for mf in mask_fracs]
        
        # =================================================================
        # Panel A: Accuracy vs mask level (line plot)
        # =================================================================
        ax = axes[0]
        
        ax.plot([mf * 100 for mf in mask_fracs], accuracies, 'o-',
               color='#336699', linewidth=2, markersize=10)
        ax.axhline(y=1/256, color='red', linestyle='--', alpha=0.5,
                  label='Random (1/256)')
        
        ax.set_xlabel("Mask percentage", fontsize=11)
        ax.set_ylabel("Character accuracy", fontsize=11)
        ax.set_ylim(0, max(accuracies) * 1.2 if max(accuracies) > 0 else 0.2)
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Reconstruction Accuracy vs Masking", fontsize=12)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # =================================================================
        # Panel B: Accuracy bar chart with comparison to random
        # =================================================================
        ax = axes[1]
        
        x_pos = np.arange(len(mask_fracs))
        bar_width = 0.6
        
        # Use gradient colors based on accuracy
        colors = plt.cm.viridis([a / max(accuracies) if max(accuracies) > 0 else 0 for a in accuracies])
        
        bars = ax.bar(x_pos, accuracies, width=bar_width, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.9)
        
        # Add accuracy labels
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.1%}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Random baseline
        ax.axhline(y=1/256, color='red', linestyle='--', alpha=0.7,
                  label=f'Random baseline ({1/256:.2%})')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{mf*100:.0f}%" for mf in mask_fracs], fontsize=10)
        ax.set_xlabel("Mask level", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, max(accuracies) * 1.3 if max(accuracies) > 0 else 0.2)
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Accuracy by Mask Level", fontsize=12)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "text_diffusion.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
