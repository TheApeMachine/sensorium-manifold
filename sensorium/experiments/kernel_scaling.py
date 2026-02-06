"""Scaling analysis experiments for the Sensorium Manifold.

This experiment uses the clean composable pattern:
- Datasets: ScalingDataset for various test data
- Observers: ModeTrackingObserver, ParticleCount, ModeCount
- Projectors: ScalingTableProjector, ScalingDynamicsFigureProjector, ScalingComputeFigureProjector

Produces:
- `paper/tables/scaling_summary.tex`
- `paper/figures/scaling_dynamics.png`
- `paper/figures/scaling_compute.png`
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    ScalingDatasetConfig,
    ScalingDataset,
    ScalingTestType,
    GeneralizationType,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    ModeTrackingObserver,
    ParticleCount,
    ModeCount,
)

# 3. MANIFOLD
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    CoherenceSimulationConfig,
    GeometricSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig

# 4. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.scaling import (
    ScalingTableProjector,
    ScalingDynamicsFigureProjector,
    ScalingComputeFigureProjector,
)


class KernelScaling(Experiment):
    """Comprehensive scaling analysis experiment.
    
    Clean pattern:
    - datasets: ScalingDataset for each test type
    - manifold: Multiple runs with different configurations
    - inference: InferenceObserver accumulates all results
    - projector: ScalingTableProjector, ScalingDynamicsFigureProjector, ScalingComputeFigureProjector
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        self.vocab_size = 4096
        self.prime = 31
        
        # Scaling parameters to test
        self.particle_counts = [100, 500, 1000, 2000, 5000, 10000]
        self.grid_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        self.pattern_counts = [1, 2, 4, 8, 16, 32]
        self.sequence_lengths = [500, 1000, 2000, 4000, 8000]
        
        # 2. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 3. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            ScalingTableProjector(output_dir=Path("paper/tables")),
            ScalingDynamicsFigureProjector(output_dir=Path("paper/figures")),
            ScalingComputeFigureProjector(output_dir=Path("paper/figures")),
        )
    
    def run(self):
        print("[scaling] Starting scaling analysis...")
        
        results: Dict[str, Any] = {}
        
        # 1. Mode population dynamics
        print("\n" + "="*60)
        print("1. MODE POPULATION DYNAMICS")
        print("="*60)
        results["population"] = self._run_population_dynamics()
        
        # 2. Mode interference
        print("\n" + "="*60)
        print("2. MODE INTERFERENCE AT SCALE")
        print("="*60)
        results["interference"] = self._run_interference_test()
        
        # 3. Compute scaling
        print("\n" + "="*60)
        print("3. COMPUTE SCALING CURVE")
        print("="*60)
        results["compute"] = self._run_compute_scaling()
        
        # 3b. Sequence length latency (O(k) test)
        print("\n" + "="*60)
        print("3b. O(k) LATENCY TEST (sequence length independence)")
        print("="*60)
        results["latency"] = self._run_latency_test()
        
        # 4. Generalization test
        print("\n" + "="*60)
        print("4. GENERALIZATION VS MEMORIZATION")
        print("="*60)
        results["generalization"] = self._run_generalization_test()
        
        # Accumulate all results to inference observer
        self.inference.observe({}, **results)
        
        # Project
        self.project()
        
        print("\n[scaling] Experiment complete.")
    
    def _run_population_dynamics(self) -> Dict[str, Any]:
        """Track mode birth/death/stabilization over time."""
        dataset = ScalingDataset(ScalingDatasetConfig(
            test_type=ScalingTestType.POPULATION,
            n_bytes=2000,
        ))
        
        observer = ModeTrackingObserver(max_steps=300)
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                generator=dataset.generate,
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                geometric=GeometricSimulationConfig(grid_size=(32, 32, 32)),
                coherence=CoherenceSimulationConfig(
                    max_carriers=64,
                    grid_size=(32, 32, 32),
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                    volatile_decay_mul=0.90,
                    stable_decay_mul=0.98,
                ),
            ),
            observers={"coherence": observer},
        )
        
        start = time.time()
        state = manifold.run()
        wall_time = (time.time() - start) * 1000
        
        history = observer.history
        n_modes_final = history["n_modes"][-1] if history["n_modes"] else 0
        n_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
        
        total_births = sum(history["n_births"])
        total_deaths = sum(history["n_deaths"])
        pruning_rate = total_deaths / (total_births + 1)
        
        stable_modes = list(history["n_modes"][-50:])
        carrying_capacity = np.mean(stable_modes) / 64 if stable_modes else 0
        
        result = {
            "history": history,
            "n_particles": len(dataset),
            "n_modes_final": n_modes_final,
            "n_crystallized": n_crystallized,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "pruning_rate": pruning_rate,
            "carrying_capacity": carrying_capacity,
            "wall_time_ms": wall_time,
            "steps": observer.step_count,
        }
        
        print(f"  Particles: {len(dataset):,}")
        print(f"  Steps: {observer.step_count}")
        print(f"  Final modes: {n_modes_final} (crystallized: {n_crystallized})")
        print(f"  Total births: {total_births}, deaths: {total_deaths}")
        print(f"  Pruning rate: {pruning_rate:.2f}")
        print(f"  Carrying capacity: {carrying_capacity:.1%} of max")
        
        return result
    
    def _run_interference_test(self) -> list:
        """Test mode interference as pattern count increases."""
        results = []
        
        for n_patterns in self.pattern_counts:
            dataset = ScalingDataset(ScalingDatasetConfig(
                test_type=ScalingTestType.INTERFERENCE,
                n_bytes=2000,
                n_patterns=n_patterns,
            ))
            
            observer = ModeTrackingObserver(max_steps=200)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(grid_size=(32, 32, 32)),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=(32, 32, 32),
                        stable_amp_threshold=0.15,
                        crystallize_amp_threshold=0.20,
                    ),
                ),
                observers={"coherence": observer},
            )
            
            state = manifold.run()
            
            history = observer.history
            final_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
            avg_conflict = float(np.mean(history["conflict_score"])) if history["conflict_score"] else 0
            crystallization_efficiency = final_crystallized / n_patterns if n_patterns > 0 else 0
            
            result = {
                "n_patterns": n_patterns,
                "n_crystallized": final_crystallized,
                "avg_conflict": avg_conflict,
                "crystallization_efficiency": crystallization_efficiency,
                "n_bytes": len(dataset),
            }
            results.append(result)
            
            print(f"  {n_patterns} patterns: {final_crystallized} crystallized, "
                  f"conflict={avg_conflict:.3f}, efficiency={crystallization_efficiency:.1%}")
        
        return results
    
    def _run_compute_scaling(self) -> Dict[str, list]:
        """Measure wall-clock time vs various scaling factors."""
        by_particles = []
        by_grid = []
        n_steps = 50
        
        print("  Testing particle count scaling...")
        for n_particles in self.particle_counts:
            dataset = ScalingDataset(ScalingDatasetConfig(
                test_type=ScalingTestType.COMPUTE,
                n_bytes=n_particles,
                seed=42,
            ))
            
            observer = ModeTrackingObserver(max_steps=n_steps)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(grid_size=(32, 32, 32)),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=(32, 32, 32),
                    ),
                ),
                observers={"coherence": observer},
            )
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            by_particles.append({
                "n_particles": n_particles,
                "wall_time_ms": wall_time,
                "ms_per_particle": wall_time / n_particles,
                "steps": observer.step_count,
            })
            
            print(f"    {n_particles:,} particles: {wall_time:.1f}ms "
                  f"({wall_time/n_particles:.3f} ms/particle, {observer.step_count} steps)")
        
        print("  Testing grid size scaling...")
        for grid_size in self.grid_sizes:
            dataset = ScalingDataset(ScalingDatasetConfig(
                test_type=ScalingTestType.COMPUTE,
                n_bytes=2000,
                seed=42,
            ))
            
            observer = ModeTrackingObserver(max_steps=n_steps)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(grid_size=grid_size),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=grid_size,
                    ),
                ),
                observers={"coherence": observer},
            )
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            grid_cells = grid_size[0] * grid_size[1] * grid_size[2]
            
            by_grid.append({
                "grid_size": grid_size,
                "grid_cells": grid_cells,
                "wall_time_ms": wall_time,
                "steps": observer.step_count,
            })
            
            print(f"    Grid {grid_size}: {wall_time:.1f}ms ({grid_cells:,} cells, {observer.step_count} steps)")
        
        return {"by_particles": by_particles, "by_grid": by_grid}
    
    def _run_latency_test(self) -> list:
        """Test O(k) latency claim."""
        results = []
        n_steps = 20
        
        print("  Testing latency vs sequence length (fixed k)...")
        for seq_len in self.sequence_lengths:
            dataset = ScalingDataset(ScalingDatasetConfig(
                test_type=ScalingTestType.LATENCY,
                n_bytes=seq_len,
                seed=42,
            ))
            
            observer = ModeTrackingObserver(max_steps=n_steps)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(grid_size=(32, 32, 32)),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=(32, 32, 32),
                    ),
                ),
                observers={"coherence": observer},
            )
            
            start = time.time()
            state = manifold.run()
            wall_time = (time.time() - start) * 1000
            
            ms_per_step = wall_time / observer.step_count if observer.step_count > 0 else 0
            
            results.append({
                "seq_len": seq_len,
                "wall_time_ms": wall_time,
                "steps": observer.step_count,
                "ms_per_step": ms_per_step,
            })
            
            print(f"    N={seq_len:,}: {wall_time:.1f}ms total, {ms_per_step:.2f}ms/step")
        
        return results
    
    def _run_generalization_test(self) -> list:
        """Test structure emergence on natural vs synthetic data."""
        results = []
        
        test_cases = [
            ("repetitive", GeneralizationType.REPETITIVE),
            ("semi_random", GeneralizationType.SEMI_RANDOM),
            ("natural_like", GeneralizationType.NATURAL_LIKE),
            ("pure_random", GeneralizationType.PURE_RANDOM),
        ]
        
        for name, gen_type in test_cases:
            dataset = ScalingDataset(ScalingDatasetConfig(
                test_type=ScalingTestType.GENERALIZATION,
                generalization_type=gen_type,
                n_bytes=2000,
                seed=42,
            ))
            
            observer = ModeTrackingObserver(max_steps=200)
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    generator=dataset.generate,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    geometric=GeometricSimulationConfig(grid_size=(32, 32, 32)),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        grid_size=(32, 32, 32),
                        stable_amp_threshold=0.15,
                        crystallize_amp_threshold=0.20,
                    ),
                ),
                observers={"coherence": observer},
            )
            
            state = manifold.run()
            
            history = observer.history
            
            n_crystallized = history["n_crystallized"][-1] if history["n_crystallized"] else 0
            n_stable = history["n_stable"][-1] if history["n_stable"] else 0
            n_volatile = history["n_volatile"][-1] if history["n_volatile"] else 0
            
            token_ids = state.get("token_ids")
            if token_ids is not None:
                tid_np = token_ids.cpu().numpy()
                unique, counts = np.unique(tid_np, return_counts=True)
                probs = counts / counts.sum()
                entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
                max_entropy = float(np.log2(len(unique))) if len(unique) > 0 else 0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                collision_ratio = float(len(tid_np) / len(unique)) if len(unique) > 0 else 0
            else:
                normalized_entropy = 1.0
                collision_ratio = 1.0
            
            result = {
                "name": name,
                "n_bytes": len(dataset),
                "n_crystallized": n_crystallized,
                "n_stable": n_stable,
                "n_volatile": n_volatile,
                "structure_ratio": n_crystallized / 64,
                "normalized_entropy": normalized_entropy,
                "collision_ratio": collision_ratio,
            }
            results.append(result)
            
            print(f"  {name}: crystallized={n_crystallized}, stable={n_stable}, "
                  f"entropy={normalized_entropy:.2f}, collision={collision_ratio:.1f}x")
        
        return results
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
    
    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
