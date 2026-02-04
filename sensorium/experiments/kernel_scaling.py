"""Scaling analysis experiments for the Sensorium Manifold.

This experiment suite answers four critical scaling questions:

1. CARRIER POPULATION DYNAMICS
   - Does carrier count stabilize or grow unboundedly?
   - How aggressive is metabolic pruning?
   - What's the "carrying capacity" of a given manifold size?

2. MODE INTERFERENCE AT SCALE
   - Do carriers collide/interfere as patterns increase?
   - Can conflict-driven splitting keep up?
   - At what point does it fragment into noise?

3. COMPUTE SCALING CURVE
   - Empirical O(k) latency vs sequence length
   - Wall-clock time vs particle count, carrier count, grid resolution

4. GENERALIZATION VS MEMORIZATION
   - Structure emergence on natural text
   - Real image patterns vs synthetic
   - When does pattern learning fail?

Produces:
- `paper/tables/scaling_summary.tex`
- `paper/figures/scaling_dynamics.png`
- `paper/figures/scaling_compute.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    SpectralSimulationConfig,
    GeometricSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.base import ObserverProtocol


class CarrierTrackingObserver(ObserverProtocol):
    """Observer that tracks carrier dynamics over time."""
    
    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.step_count = 0
        
        # Time series data
        self.history = {
            "step": [],
            "n_carriers": [],
            "n_volatile": [],
            "n_stable": [],
            "n_crystallized": [],
            "max_amplitude": [],
            "mean_amplitude": [],
            "n_births": [],  # New carriers this step
            "n_deaths": [],  # Pruned carriers this step
            "conflict_score": [],
        }
        self._prev_carrier_count = 0
    
    def observe(self, observation=None, **kwargs):
        self.step_count += 1
        
        if observation is None:
            return {"done_thinking": self.step_count >= self.max_steps}
        
        # Extract carrier info
        amplitudes = observation.get("amplitudes")
        carrier_state = observation.get("carrier_state")
        conflict = observation.get("conflict")
        
        if amplitudes is not None:
            active_mask = amplitudes > 1e-6
            n_carriers = int(active_mask.sum().item())
            max_amp = float(amplitudes.max().item()) if n_carriers > 0 else 0.0
            mean_amp = float(amplitudes[active_mask].mean().item()) if n_carriers > 0 else 0.0
        else:
            n_carriers = 0
            max_amp = 0.0
            mean_amp = 0.0
        
        # Count states: 0=volatile, 1=stable, 2=crystallized
        n_volatile = n_stable = n_crystallized = 0
        if carrier_state is not None and n_carriers > 0:
            states = carrier_state[:n_carriers]
            n_volatile = int((states == 0).sum().item())
            n_stable = int((states == 1).sum().item())
            n_crystallized = int((states == 2).sum().item())
        
        # Births and deaths
        n_births = max(0, n_carriers - self._prev_carrier_count)
        n_deaths = max(0, self._prev_carrier_count - n_carriers)
        self._prev_carrier_count = n_carriers
        
        # Conflict score
        conflict_score = 0.0
        if conflict is not None and n_carriers > 0:
            conflict_score = float(conflict[:n_carriers].mean().item())
        
        # Record
        self.history["step"].append(self.step_count)
        self.history["n_carriers"].append(n_carriers)
        self.history["n_volatile"].append(n_volatile)
        self.history["n_stable"].append(n_stable)
        self.history["n_crystallized"].append(n_crystallized)
        self.history["max_amplitude"].append(max_amp)
        self.history["mean_amplitude"].append(mean_amp)
        self.history["n_births"].append(n_births)
        self.history["n_deaths"].append(n_deaths)
        self.history["conflict_score"].append(conflict_score)
        
        return {"done_thinking": self.step_count >= self.max_steps}


@dataclass
class ScalingResult:
    """Container for scaling experiment results."""
    n_particles: int
    n_carriers_final: int
    n_crystallized: int
    wall_time_ms: float
    steps: int
    grid_size: Tuple[int, int, int]
    carrier_history: Dict[str, List]
    carrying_capacity_ratio: float  # carriers / max_carriers
    pruning_rate: float  # deaths / (births + 1)


class KernelScaling(Experiment):
    """Comprehensive scaling analysis experiment."""
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        self.vocab_size = 4096
        self.prime = 31
        
        # Scaling parameters to test
        self.particle_counts = [100, 500, 1000, 2000, 5000, 10000]
        self.grid_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        self.pattern_counts = [1, 2, 4, 8, 16, 32]  # For interference test
        self.sequence_lengths = [500, 1000, 2000, 4000, 8000]  # For O(k) latency test
        
        self.results: Dict[str, Any] = {}
    
    def run(self):
        print("[scaling] Starting scaling analysis...")
        
        # 1. Carrier population dynamics
        print("\n" + "="*60)
        print("1. CARRIER POPULATION DYNAMICS")
        print("="*60)
        self._run_population_dynamics()
        
        # 2. Mode interference
        print("\n" + "="*60)
        print("2. MODE INTERFERENCE AT SCALE")
        print("="*60)
        self._run_interference_test()
        
        # 3. Compute scaling
        print("\n" + "="*60)
        print("3. COMPUTE SCALING CURVE")
        print("="*60)
        self._run_compute_scaling()
        
        # 3b. Sequence length latency (O(k) test)
        print("\n" + "="*60)
        print("3b. O(k) LATENCY TEST (sequence length independence)")
        print("="*60)
        self._run_latency_test()
        
        # 4. Generalization test
        print("\n" + "="*60)
        print("4. GENERALIZATION VS MEMORIZATION")
        print("="*60)
        self._run_generalization_test()
        
        # Generate artifacts
        self._generate_tables()
        self._generate_figures()
        
        print("\n[scaling] Experiment complete.")
    
    def _run_population_dynamics(self):
        """Track carrier birth/death/stabilization over time."""
        
        # Use medium-sized data stream
        n_bytes = 2000
        text = ("The quick brown fox jumps over the lazy dog. " * 50)[:n_bytes]
        data = text.encode("utf-8")
        
        def data_generator(d=data):
            yield d
        
        observer = CarrierTrackingObserver(max_steps=300)
        
        cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=self.vocab_size,
                hash_prime=self.prime,
                segment_size=64,
            ),
            geometric=GeometricSimulationConfig(
                grid_size=(32, 32, 32),
            ),
            spectral=SpectralSimulationConfig(
                max_carriers=64,
                grid_size=(32, 32, 32),
                stable_amp_threshold=0.15,
                crystallize_amp_threshold=0.20,
                volatile_decay_mul=0.90,
                stable_decay_mul=0.98,
            ),
            generator=data_generator,
        )
        
        manifold = Manifold(cfg, observers={"spectral": observer})
        
        start = time.time()
        state = manifold.run()
        wall_time = (time.time() - start) * 1000
        
        # Calculate summary metrics
        history = observer.history
        n_carriers_final = history["n_carriers"][-1] if history["n_carriers"] else 0
        n_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
        
        total_births = sum(history["n_births"])
        total_deaths = sum(history["n_deaths"])
        pruning_rate = total_deaths / (total_births + 1)
        
        # Carrying capacity: stable carrier count / max_carriers
        stable_carriers = [n for n in history["n_carriers"][-50:]]  # Last 50 steps
        carrying_capacity = np.mean(stable_carriers) / 64 if stable_carriers else 0
        
        self.results["population"] = {
            "history": history,
            "n_particles": len(data),
            "n_carriers_final": n_carriers_final,
            "n_crystallized": n_crystallized,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "pruning_rate": pruning_rate,
            "carrying_capacity": carrying_capacity,
            "wall_time_ms": wall_time,
            "steps": observer.step_count,
        }
        
        print(f"  Particles: {len(data):,}")
        print(f"  Steps: {observer.step_count}")
        print(f"  Final carriers: {n_carriers_final} (crystallized: {n_crystallized})")
        print(f"  Total births: {total_births}, deaths: {total_deaths}")
        print(f"  Pruning rate: {pruning_rate:.2f}")
        print(f"  Carrying capacity: {carrying_capacity:.1%} of max")
    
    def _run_interference_test(self):
        """Test carrier interference as pattern count increases."""
        
        self.results["interference"] = []
        
        for n_patterns in self.pattern_counts:
            # Create data with N distinct patterns
            patterns = [f"Pattern{i:02d}XYZ" for i in range(n_patterns)]
            text = " ".join(patterns * (200 // n_patterns))
            data = text.encode("utf-8")[:2000]
            
            def data_generator(d=data):
                yield d
            
            observer = CarrierTrackingObserver(max_steps=200)
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                    segment_size=32,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    grid_size=(32, 32, 32),
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                ),
                generator=data_generator,
            )
            
            manifold = Manifold(cfg, observers={"spectral": observer})
            state = manifold.run()
            
            history = observer.history
            
            # Measure interference
            final_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
            avg_conflict = np.mean(history["conflict_score"]) if history["conflict_score"] else 0
            
            # Expected: crystallized should roughly equal n_patterns (up to a point)
            crystallization_efficiency = final_crystallized / n_patterns if n_patterns > 0 else 0
            
            result = {
                "n_patterns": n_patterns,
                "n_crystallized": final_crystallized,
                "avg_conflict": avg_conflict,
                "crystallization_efficiency": crystallization_efficiency,
                "n_bytes": len(data),
            }
            self.results["interference"].append(result)
            
            print(f"  {n_patterns} patterns: {final_crystallized} crystallized, "
                  f"conflict={avg_conflict:.3f}, efficiency={crystallization_efficiency:.1%}")
    
    def _run_compute_scaling(self):
        """Measure wall-clock time vs various scaling factors."""
        
        self.results["compute"] = {
            "by_particles": [],
            "by_grid": [],
        }
        
        # Fixed number of simulation steps for consistent comparison
        n_steps = 50
        
        # Scaling by particle count (fixed grid)
        print("  Testing particle count scaling...")
        for n_particles in self.particle_counts:
            # Use reproducible random data
            np.random.seed(42)
            data = bytes(np.random.randint(0, 256, n_particles, dtype=np.uint8))
            
            def data_generator(d=data):
                yield d
            
            observer = CarrierTrackingObserver(max_steps=n_steps)
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    grid_size=(32, 32, 32),
                ),
                generator=data_generator,
            )
            
            manifold = Manifold(cfg, observers={"spectral": observer})
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            self.results["compute"]["by_particles"].append({
                "n_particles": n_particles,
                "wall_time_ms": wall_time,
                "ms_per_particle": wall_time / n_particles,
                "steps": observer.step_count,
            })
            
            print(f"    {n_particles:,} particles: {wall_time:.1f}ms "
                  f"({wall_time/n_particles:.3f} ms/particle, {observer.step_count} steps)")
        
        # Scaling by grid size (fixed particles)
        print("  Testing grid size scaling...")
        n_particles = 2000
        np.random.seed(42)
        data = bytes(np.random.randint(0, 256, n_particles, dtype=np.uint8))
        
        for grid_size in self.grid_sizes:
            def data_generator(d=data):
                yield d
            
            observer = CarrierTrackingObserver(max_steps=n_steps)
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=grid_size,
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    grid_size=grid_size,
                ),
                generator=data_generator,
            )
            
            manifold = Manifold(cfg, observers={"spectral": observer})
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            grid_cells = grid_size[0] * grid_size[1] * grid_size[2]
            
            self.results["compute"]["by_grid"].append({
                "grid_size": grid_size,
                "grid_cells": grid_cells,
                "wall_time_ms": wall_time,
                "steps": observer.step_count,
            })
            
            print(f"    Grid {grid_size}: {wall_time:.1f}ms ({grid_cells:,} cells, {observer.step_count} steps)")
    
    def _run_latency_test(self):
        """Test O(k) latency claim: step time should be independent of sequence length.
        
        The claim is that manifold latency scales with k (active carriers), not N (sequence length).
        We test this by varying sequence length while keeping max_carriers fixed.
        """
        
        self.results["latency"] = []
        n_steps = 20  # Fewer steps, focus on per-step timing
        
        print("  Testing latency vs sequence length (fixed k)...")
        for seq_len in self.sequence_lengths:
            # Create data of varying lengths
            np.random.seed(42)
            data = bytes(np.random.randint(0, 256, seq_len, dtype=np.uint8))
            
            def data_generator(d=data):
                yield d
            
            observer = CarrierTrackingObserver(max_steps=n_steps)
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,  # Fixed k
                    grid_size=(32, 32, 32),
                ),
                generator=data_generator,
            )
            
            manifold = Manifold(cfg, observers={"spectral": observer})
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            ms_per_step = wall_time / observer.step_count if observer.step_count > 0 else 0
            
            self.results["latency"].append({
                "seq_len": seq_len,
                "wall_time_ms": wall_time,
                "steps": observer.step_count,
                "ms_per_step": ms_per_step,
            })
            
            print(f"    N={seq_len:,}: {wall_time:.1f}ms total, {ms_per_step:.2f}ms/step")
    
    def _run_generalization_test(self):
        """Test structure emergence on natural vs synthetic data."""
        
        self.results["generalization"] = []
        
        # Test cases: synthetic repetitive, synthetic random, natural-ish
        test_cases = [
            ("repetitive", "The cat sat on the mat. " * 100),
            ("semi_random", "".join(chr(65 + (i * 7) % 26) for i in range(2000))),
            ("natural_like", self._get_natural_text()),
            ("pure_random", "".join(chr(np.random.randint(32, 127)) for _ in range(2000))),
        ]
        
        for name, text in test_cases:
            data = text.encode("utf-8")[:2000]
            
            def data_generator(d=data):
                yield d
            
            observer = CarrierTrackingObserver(max_steps=200)
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                    segment_size=64,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    grid_size=(32, 32, 32),
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                ),
                generator=data_generator,
            )
            
            manifold = Manifold(cfg, observers={"spectral": observer})
            state = manifold.run()
            
            history = observer.history
            
            # Structure metrics
            n_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
            n_stable = history["n_stable"][-1] if history["n_stable"] else 0
            n_volatile = history["n_volatile"][-1] if history["n_volatile"] else 0
            
            # Entropy of token IDs (measure of pattern regularity)
            token_ids = state.get("token_ids")
            if token_ids is not None:
                tid_np = token_ids.cpu().numpy()
                unique, counts = np.unique(tid_np, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(unique)) if len(unique) > 0 else 0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                collision_ratio = len(tid_np) / len(unique) if len(unique) > 0 else 0
            else:
                normalized_entropy = 1.0
                collision_ratio = 1.0
            
            result = {
                "name": name,
                "n_bytes": len(data),
                "n_crystallized": n_crystallized,
                "n_stable": n_stable,
                "n_volatile": n_volatile,
                "structure_ratio": n_crystallized / 64,  # How much structure emerged
                "normalized_entropy": normalized_entropy,
                "collision_ratio": collision_ratio,
            }
            self.results["generalization"].append(result)
            
            print(f"  {name}: crystallized={n_crystallized}, stable={n_stable}, "
                  f"entropy={normalized_entropy:.2f}, collision={collision_ratio:.1f}x")
    
    def _get_natural_text(self) -> str:
        """Get natural-ish text (varied sentence structures)."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The only thing we have to fear is fear itself.",
            "In the beginning was the word.",
            "It was the best of times, it was the worst of times.",
            "Ask not what your country can do for you.",
            "I think, therefore I am.",
            "The truth shall set you free.",
        ]
        # Mix sentences with some repetition but not pure repetition
        text = ""
        for i in range(50):
            text += sentences[i % len(sentences)] + " "
            if i % 3 == 0:
                text += sentences[(i * 7) % len(sentences)] + " "
        return text
    
    def _generate_tables(self):
        """Generate LaTeX summary tables."""
        
        # Main scaling summary table
        pop = self.results.get("population", {})
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Scaling analysis summary. Carrier population dynamics show the manifold's ``carrying capacity'' and metabolic pruning behavior. Interference results show crystallization efficiency as pattern count increases.}
\label{tab:scaling}
\begin{tabular}{l r}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Population Dynamics}} \\
\quad Final carriers & """ + str(pop.get("n_carriers_final", 0)) + r""" \\
\quad Crystallized & """ + str(pop.get("n_crystallized", 0)) + r""" \\
\quad Total births & """ + str(pop.get("total_births", 0)) + r""" \\
\quad Total deaths & """ + str(pop.get("total_deaths", 0)) + r""" \\
\quad Pruning rate & """ + f"{pop.get('pruning_rate', 0):.2f}" + r""" \\
\quad Carrying capacity & """ + f"{pop.get('carrying_capacity', 0)*100:.1f}\\%" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Compute Scaling}} \\
"""
        
        compute = self.results.get("compute", {})
        if compute.get("by_particles"):
            first = compute["by_particles"][0]
            last = compute["by_particles"][-1]
            table_content += f"\\quad {first['n_particles']:,} particles & {first['wall_time_ms']:.0f} ms \\\\\n"
            table_content += f"\\quad {last['n_particles']:,} particles & {last['wall_time_ms']:.0f} ms \\\\\n"
        
        # Latency test (O(k) claim)
        latency = self.results.get("latency", [])
        if latency:
            table_content += r"\midrule" + "\n"
            table_content += r"\multicolumn{2}{l}{\textit{Latency vs Sequence Length (O(k) test)}} \\" + "\n"
            for lat in latency:
                table_content += f"\\quad N={lat['seq_len']:,} & {lat['ms_per_step']:.2f} ms/step \\\\\n"
        
        table_content += r"""\midrule
\multicolumn{2}{l}{\textit{Generalization}} \\
"""
        
        gen = self.results.get("generalization", [])
        for g in gen:
            table_content += f"\\quad {g['name'].replace('_', ' ').title()} & {g['n_crystallized']} crystallized \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "scaling_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def _generate_figures(self):
        """Generate scaling visualization figures."""
        import matplotlib.pyplot as plt
        
        # Figure 1: Population dynamics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        pop = self.results.get("population", {})
        history = pop.get("history", {})
        
        if history:
            steps = history["step"]
            
            # Panel A: Carrier population over time
            ax = axes[0, 0]
            ax.plot(steps, history["n_carriers"], label="Total", color='#336699', linewidth=2)
            ax.plot(steps, history["n_crystallized"], label="Crystallized", color='#27ae60', linewidth=2)
            ax.plot(steps, history["n_stable"], label="Stable", color='#f39c12', linewidth=2, linestyle='--')
            ax.plot(steps, history["n_volatile"], label="Volatile", color='#e74c3c', linewidth=2, linestyle=':')
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel("Carrier count", fontsize=10)
            ax.legend(loc='right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
            
            # Panel B: Birth/death rates
            ax = axes[0, 1]
            ax.bar(steps, history["n_births"], alpha=0.7, color='#27ae60', label="Births", width=1)
            ax.bar(steps, [-d for d in history["n_deaths"]], alpha=0.7, color='#e74c3c', label="Deaths", width=1)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel("Birth/Death count", fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Interference test
        ax = axes[1, 0]
        interference = self.results.get("interference", [])
        if interference:
            n_patterns = [r["n_patterns"] for r in interference]
            n_crystallized = [r["n_crystallized"] for r in interference]
            efficiency = [r["crystallization_efficiency"] for r in interference]
            
            ax.plot(n_patterns, n_crystallized, 'o-', color='#336699', linewidth=2, markersize=8, label="Crystallized")
            ax.plot(n_patterns, n_patterns, '--', color='gray', linewidth=1, alpha=0.5, label="Ideal (1:1)")
            ax.set_xlabel("Number of distinct patterns", fontsize=10)
            ax.set_ylabel("Crystallized carriers", fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Panel D: Generalization
        ax = axes[1, 1]
        gen = self.results.get("generalization", [])
        if gen:
            names = [g["name"].replace("_", "\n") for g in gen]
            structure = [g["structure_ratio"] for g in gen]
            colors = ['#27ae60' if s > 0.3 else '#f39c12' if s > 0.1 else '#e74c3c' for s in structure]
            
            bars = ax.bar(range(len(names)), structure, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel("Structure ratio (crystallized / max)", fontsize=10)
            ax.set_ylim(0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "scaling_dynamics.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        
        # Figure 2: Compute scaling (3 panels)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        compute = self.results.get("compute", {})
        
        # Panel A: Time vs particle count
        ax = axes[0]
        by_particles = compute.get("by_particles", [])
        if by_particles:
            particles = [r["n_particles"] for r in by_particles]
            times = [r["wall_time_ms"] for r in by_particles]
            
            ax.plot(particles, times, 'o-', color='#336699', linewidth=2, markersize=8)
            ax.set_xlabel("Particle count", fontsize=10)
            ax.set_ylabel("Wall-clock time (ms)", fontsize=10)
            
            # Fit line to check scaling
            if len(particles) > 2:
                z = np.polyfit(particles, times, 1)
                p = np.poly1d(z)
                ax.plot(particles, p(particles), '--', color='gray', alpha=0.5, 
                       label=f"Linear fit: {z[0]:.4f}ms/particle")
                ax.legend(loc='upper left', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel B: Time vs grid size
        ax = axes[1]
        by_grid = compute.get("by_grid", [])
        if by_grid:
            cells = [r["grid_cells"] for r in by_grid]
            times = [r["wall_time_ms"] for r in by_grid]
            labels = [f"{r['grid_size'][0]}³" for r in by_grid]
            
            ax.bar(range(len(cells)), times, color='#4C994C', edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_xlabel("Grid size", fontsize=10)
            ax.set_ylabel("Wall-clock time (ms)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        # Panel C: Latency vs sequence length (O(k) test)
        ax = axes[2]
        latency = self.results.get("latency", [])
        if latency:
            seq_lens = [r["seq_len"] for r in latency]
            ms_per_step = [r["ms_per_step"] for r in latency]
            
            ax.plot(seq_lens, ms_per_step, 'o-', color='#9b59b6', linewidth=2, markersize=8)
            ax.set_xlabel("Sequence length N", fontsize=10)
            ax.set_ylabel("Latency (ms/step)", fontsize=10)
            
            # Show mean line
            mean_latency = np.mean(ms_per_step)
            ax.axhline(y=mean_latency, color='gray', linestyle='--', alpha=0.5,
                      label=f"Mean: {mean_latency:.2f} ms/step")
            ax.legend(loc='upper left', fontsize=9)
            
            # Calculate variance to show O(k) claim
            std_latency = np.std(ms_per_step)
            cv = std_latency / mean_latency if mean_latency > 0 else 0
            ax.set_title(f"O(k) test: CV={cv:.1%}", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "scaling_compute.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
