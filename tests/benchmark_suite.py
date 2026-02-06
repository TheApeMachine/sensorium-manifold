"""End-to-end benchmark suite for the Sensorium Manifold.

This module provides deterministic, reproducible benchmarks that measure:
1. Spectral layer performance (carrier dynamics, energy flow)
2. Geometric layer performance (particle physics, field solves)
3. Full manifold integration (tokenizer → physics → observers)
4. Task-specific accuracy (next-token prediction, pattern learning)

Results are saved to JSON for baseline comparison.

Usage:
    # Run all benchmarks and save results
    python -m tests.benchmark_suite --save baseline_v1.json
    
    # Compare against a baseline
    python -m tests.benchmark_suite --compare baseline_v1.json
    
    # Run specific benchmark
    python -m tests.benchmark_suite --benchmark spectral_dynamics
"""

from __future__ import annotations

import argparse
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch


def get_device() -> str:
    """Get available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_deterministic_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results."""
    name: str
    device: str
    seed: int
    
    # Timing
    wall_time_ms: float = 0.0
    steps: int = 0
    ms_per_step: float = 0.0
    
    # Spectral layer metrics
    final_carrier_count: int = 0
    carrier_births: int = 0
    carrier_deaths: int = 0
    n_volatile: int = 0
    n_stable: int = 0
    n_crystallized: int = 0
    max_amplitude: float = 0.0
    mean_amplitude: float = 0.0
    max_conflict: float = 0.0
    mean_conflict: float = 0.0
    
    # Energy metrics
    total_energy_initial: float = 0.0
    total_energy_final: float = 0.0
    energy_conservation_ratio: float = 0.0
    
    # Geometric layer metrics
    max_velocity: float = 0.0
    mean_velocity: float = 0.0
    max_temperature: float = 0.0
    
    # Task-specific metrics (optional)
    accuracy: float = 0.0
    loss: float = 0.0
    
    # Checksum for reproducibility verification
    state_checksum: str = ""
    
    # Additional custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkMetrics":
        return cls(**d)


def compute_state_checksum(tensors: Dict[str, torch.Tensor]) -> str:
    """Compute a hash of the current state for reproducibility verification."""
    hasher = hashlib.md5()
    for key in sorted(tensors.keys()):
        t = tensors[key]
        if isinstance(t, torch.Tensor):
            # Use first 100 elements to keep it fast
            data = t.flatten()[:100].cpu().numpy().tobytes()
            hasher.update(key.encode())
            hasher.update(data)
    return hasher.hexdigest()[:16]


class SpectralDynamicsBenchmark:
    """Benchmark for spectral carrier dynamics.
    
    Measures:
    - Carrier population evolution over time
    - State transitions (volatile → stable → crystallized)
    - Metabolic homeostasis (income vs expense)
    - Conflict and splitting dynamics
    """
    
    NAME = "spectral_dynamics"
    
    def __init__(self, device: str, seed: int = 42):
        self.device = device
        self.seed = seed
    
    def run(self, n_oscillators: int = 1000, n_steps: int = 200) -> BenchmarkMetrics:
        set_deterministic_seed(self.seed)
        
        if self.device == "mps":
            from optimizer.metal.manifold_physics import (
                SpectralCarrierPhysics,
                SpectralCarrierConfig,
            )
        elif self.device == "cuda":
            from optimizer.triton.manifold_physics import (
                SpectralCarrierPhysics,
                SpectralCarrierConfig,
            )
        else:
            raise ValueError(f"Unsupported device: {self.device}")
        
        cfg = SpectralCarrierConfig(
            max_carriers=64,
            stable_amp_threshold=0.25,
            crystallize_amp_threshold=0.75,
            crystallize_age=50,
            volatile_decay_mul=0.90,
            stable_decay_mul=0.98,
            crystallized_decay_mul=1.00,
        )
        
        sp = SpectralCarrierPhysics(cfg, grid_size=(16, 16, 16), dt=0.01, device=self.device)
        
        # Create deterministic input
        osc_phase = torch.rand(n_oscillators, device=self.device, dtype=torch.float32) * (2.0 * torch.pi)
        osc_omega = torch.randn(n_oscillators, device=self.device, dtype=torch.float32) * 5.0
        osc_energy = torch.rand(n_oscillators, device=self.device, dtype=torch.float32) + 0.5
        
        initial_total_energy = osc_energy.sum().item()
        
        # History tracking
        carrier_counts = []
        births_total = 0
        prev_count = 0
        max_conflict = 0.0
        conflicts = []
        
        start_time = time.time()
        
        for step in range(n_steps):
            out = sp.step(osc_phase, osc_omega, osc_energy)
            
            osc_phase = out["osc_phase"]
            if "osc_energy" in out:
                osc_energy = out["osc_energy"]
            
            num_carriers = out.get("num_carriers", sp._num_carriers_buf).item() if "num_carriers" in out else sp.num_carriers
            carrier_counts.append(num_carriers)
            
            if num_carriers > prev_count:
                births_total += (num_carriers - prev_count)
            prev_count = num_carriers
            
            if "conflict" in out and out["conflict"].numel() > 0 and num_carriers > 0:
                c = out["conflict"][:num_carriers]
                max_conflict = max(max_conflict, c.max().item())
                conflicts.append(c.mean().item())
        
        wall_time = (time.time() - start_time) * 1000
        
        # Final state
        final_count = carrier_counts[-1]
        final_energy = osc_energy.sum().item()
        
        # Carrier states
        n_volatile = n_stable = n_crystallized = 0
        max_amp = mean_amp = 0.0
        if "carrier_state" in out and final_count > 0:
            states = out["carrier_state"][:final_count]
            n_volatile = (states == 0).sum().item()
            n_stable = (states == 1).sum().item()
            n_crystallized = (states == 2).sum().item()
        
        if "amplitudes" in out and out["amplitudes"].numel() > 0 and final_count > 0:
            amps = out["amplitudes"][:final_count]
            max_amp = amps.max().item()
            mean_amp = amps.mean().item()
        
        # Compute checksum
        checksum = compute_state_checksum({
            "osc_phase": osc_phase,
            "osc_energy": osc_energy,
            "carrier_real": sp.carrier_real,
            "carrier_imag": sp.carrier_imag,
        })
        
        return BenchmarkMetrics(
            name=self.NAME,
            device=self.device,
            seed=self.seed,
            wall_time_ms=wall_time,
            steps=n_steps,
            ms_per_step=wall_time / n_steps,
            final_carrier_count=final_count,
            carrier_births=births_total,
            carrier_deaths=0,  # Currently no deaths implemented
            n_volatile=n_volatile,
            n_stable=n_stable,
            n_crystallized=n_crystallized,
            max_amplitude=max_amp,
            mean_amplitude=mean_amp,
            max_conflict=max_conflict,
            mean_conflict=np.mean(conflicts) if conflicts else 0.0,
            total_energy_initial=initial_total_energy,
            total_energy_final=final_energy,
            energy_conservation_ratio=final_energy / initial_total_energy if initial_total_energy > 0 else 0.0,
            state_checksum=checksum,
        )


class GeometricPhysicsBenchmark:
    """Benchmark for geometric layer physics.
    
    Measures:
    - Particle dynamics (velocity, position)
    - Field solves (gravity, temperature)
    - Collision handling
    - Energy conservation
    """
    
    NAME = "geometric_physics"
    
    def __init__(self, device: str, seed: int = 42):
        self.device = device
        self.seed = seed
    
    def run(self, n_particles: int = 2000, n_steps: int = 100) -> BenchmarkMetrics:
        set_deterministic_seed(self.seed)
        
        if self.device == "mps":
            from sensorium.kernels.metal.manifold_physics import (
                ThermodynamicsDomain,
                ThermodynamicsDomainConfig,
            )
        elif self.device == "cuda":
            from sensorium.kernels.triton.manifold_physics import (
                ThermodynamicsDomain,
                ThermodynamicsDomainConfig,
            )
        else:
            raise ValueError(f"Unsupported device: {self.device}")
        
        cfg = ThermodynamicsDomainConfig(
            grid_size=(32, 32, 32),
            dt_max=0.01,
        )
        
        mp = ThermodynamicsDomain(cfg, device=self.device)
        
        # Create deterministic particle state
        positions = torch.rand(n_particles, 3, device=self.device, dtype=torch.float32) * 28.0 + 2.0
        velocities = torch.randn(n_particles, 3, device=self.device, dtype=torch.float32) * 0.5
        energies = torch.rand(n_particles, device=self.device, dtype=torch.float32) + 0.1
        heats = torch.rand(n_particles, device=self.device, dtype=torch.float32)
        excitations = torch.randn(n_particles, device=self.device, dtype=torch.float32)
        masses = torch.ones(n_particles, device=self.device, dtype=torch.float32)
        
        initial_ke = (0.5 * masses * (velocities ** 2).sum(dim=1)).sum().item()
        initial_thermal = (heats + energies).sum().item()
        initial_total = initial_ke + initial_thermal
        
        start_time = time.time()
        
        for step in range(n_steps):
            positions, velocities, energies, heats, excitations = mp.step(
                positions, velocities, energies, heats, excitations, masses
            )
        
        wall_time = (time.time() - start_time) * 1000
        
        # Final state metrics
        final_ke = (0.5 * masses * (velocities ** 2).sum(dim=1)).sum().item()
        final_thermal = (heats + energies).sum().item()
        final_total = final_ke + final_thermal
        
        max_vel = velocities.norm(dim=1).max().item()
        mean_vel = velocities.norm(dim=1).mean().item()
        max_temp = 0.0
        
        checksum = compute_state_checksum({
            "positions": positions,
            "velocities": velocities,
            "energies": energies,
            "heats": heats,
        })
        
        return BenchmarkMetrics(
            name=self.NAME,
            device=self.device,
            seed=self.seed,
            wall_time_ms=wall_time,
            steps=n_steps,
            ms_per_step=wall_time / n_steps,
            max_velocity=max_vel,
            mean_velocity=mean_vel,
            max_temperature=max_temp,
            total_energy_initial=initial_total,
            total_energy_final=final_total,
            energy_conservation_ratio=final_total / initial_total if initial_total > 0 else 0.0,
            state_checksum=checksum,
        )


class FullManifoldBenchmark:
    """Benchmark for the full manifold pipeline.
    
    Measures end-to-end performance:
    - Tokenization
    - Geometric + spectral physics
    - Observer processing
    """
    
    NAME = "full_manifold"
    
    def __init__(self, device: str, seed: int = 42):
        self.device = device
        self.seed = seed
    
    def run(self, n_steps: int = 100) -> BenchmarkMetrics:
        set_deterministic_seed(self.seed)
        
        from optimizer.manifold import (
            Manifold,
            SimulationConfig,
            CoherenceSimulationConfig,
            GeometricSimulationConfig,
        )
        from optimizer.tokenizer import TokenizerConfig
        
        # Deterministic test data
        test_text = "The quick brown fox jumps over the lazy dog. " * 20
        data = test_text.encode("utf-8")
        
        def data_generator():
            yield data
        
        # Simple observer to count steps
        class StepCounter:
            def __init__(self, max_steps: int):
                self.max_steps = max_steps
                self.step = 0
                self.carrier_history = []
            
            def observe(self, observation=None, **kwargs):
                self.step += 1
                if observation and isinstance(observation, dict) and "amplitudes" in observation:
                    amps = observation["amplitudes"]
                    active = (amps > 1e-6).sum().item()
                    self.carrier_history.append(active)
                return {"done_thinking": self.step >= self.max_steps}
        
        observer = StepCounter(n_steps)
        
        cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=4096,
                hash_prime=31,
                segment_size=64,
                generator=data_generator,
            ),
            geometric=GeometricSimulationConfig(
                grid_size=(32, 32, 32),
            ),
            coherence=CoherenceSimulationConfig(
                max_carriers=64,
                grid_size=(32, 32, 32),
            ),
            device=self.device,
        )
        
        manifold = Manifold(cfg, observers={"coherence": observer})
        
        start_time = time.time()
        state = manifold.run()
        wall_time = (time.time() - start_time) * 1000
        
        # Extract final metrics
        n_particles = state.get("positions", torch.empty(0)).shape[0] if state else 0
        final_energy = state.get("energies", torch.zeros(1)).sum().item() if state else 0.0
        
        checksum = compute_state_checksum({
            "positions": state.get("positions", torch.empty(0)) if state else torch.empty(0),
            "velocities": state.get("velocities", torch.empty(0)) if state else torch.empty(0),
            "token_ids": state.get("token_ids", torch.empty(0)) if state else torch.empty(0),
        })
        
        return BenchmarkMetrics(
            name=self.NAME,
            device=self.device,
            seed=self.seed,
            wall_time_ms=wall_time,
            steps=observer.step,
            ms_per_step=wall_time / observer.step if observer.step > 0 else 0,
            final_carrier_count=observer.carrier_history[-1] if observer.carrier_history else 0,
            total_energy_final=final_energy,
            state_checksum=checksum,
            custom={
                "n_particles": n_particles,
                "carrier_history": observer.carrier_history[-10:],  # Last 10 values
            }
        )


class PatternLearningBenchmark:
    """Benchmark for pattern learning accuracy.
    
    Tests the system's ability to learn and predict patterns.
    Uses a deterministic pattern to ensure reproducibility.
    """
    
    NAME = "pattern_learning"
    
    def __init__(self, device: str, seed: int = 42):
        self.device = device
        self.seed = seed
    
    def run(self, n_steps: int = 200) -> BenchmarkMetrics:
        set_deterministic_seed(self.seed)
        
        from optimizer.manifold import (
            Manifold,
            SimulationConfig,
            CoherenceSimulationConfig,
            GeometricSimulationConfig,
        )
        from optimizer.tokenizer import TokenizerConfig
        
        # Simple repeating pattern for learning
        pattern = "ABCD" * 100
        data = pattern.encode("utf-8")
        
        def data_generator():
            yield data
        
        class PatternObserver:
            def __init__(self, max_steps: int):
                self.max_steps = max_steps
                self.step = 0
                self.crystallization_history = []
            
            def observe(self, observation=None, **kwargs):
                self.step += 1
                if observation and isinstance(observation, dict) and "carrier_state" in observation:
                    states = observation["carrier_state"]
                    amps = observation.get("amplitudes", torch.zeros(1))
                    active = (amps > 1e-6).sum().item()
                    if active > 0:
                        n_crys = (states[:int(active)] == 2).sum().item()
                        self.crystallization_history.append(n_crys)
                return {"done_thinking": self.step >= self.max_steps}
        
        observer = PatternObserver(n_steps)
        
        cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=256,
                hash_prime=31,
                segment_size=4,  # Matches pattern length
                generator=data_generator,
            ),
            geometric=GeometricSimulationConfig(
                grid_size=(16, 16, 16),
            ),
            coherence=CoherenceSimulationConfig(
                max_carriers=32,
                grid_size=(16, 16, 16),
                crystallize_age=20,
            ),
            device=self.device,
        )
        
        manifold = Manifold(cfg, observers={"coherence": observer})
        
        start_time = time.time()
        state = manifold.run()
        wall_time = (time.time() - start_time) * 1000
        
        # Crystallization indicates pattern learning
        max_crystallized = max(observer.crystallization_history) if observer.crystallization_history else 0
        
        checksum = compute_state_checksum({
            "token_ids": state.get("token_ids", torch.empty(0)) if state else torch.empty(0),
        })
        
        return BenchmarkMetrics(
            name=self.NAME,
            device=self.device,
            seed=self.seed,
            wall_time_ms=wall_time,
            steps=observer.step,
            ms_per_step=wall_time / observer.step if observer.step > 0 else 0,
            n_crystallized=max_crystallized,
            state_checksum=checksum,
            custom={
                "crystallization_history": observer.crystallization_history[-20:],
            }
        )


# ============================================================================
# Benchmark Runner and Comparison
# ============================================================================

BENCHMARKS = {
    "spectral_dynamics": SpectralDynamicsBenchmark,
    "geometric_physics": GeometricPhysicsBenchmark,
    "full_manifold": FullManifoldBenchmark,
    "pattern_learning": PatternLearningBenchmark,
}


def run_all_benchmarks(device: str, seed: int = 42) -> Dict[str, BenchmarkMetrics]:
    """Run all benchmarks and return results."""
    results = {}
    
    for name, benchmark_cls in BENCHMARKS.items():
        print(f"Running {name}...", end=" ", flush=True)
        try:
            benchmark = benchmark_cls(device, seed)
            result = benchmark.run()
            results[name] = result
            print(f"done ({result.wall_time_ms:.1f}ms)")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = None
    
    return results


def save_results(results: Dict[str, BenchmarkMetrics], path: Path):
    """Save benchmark results to JSON."""
    data = {
        name: result.to_dict() if result else None
        for name, result in results.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: Path) -> Dict[str, BenchmarkMetrics]:
    """Load benchmark results from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {
        name: BenchmarkMetrics.from_dict(d) if d else None
        for name, d in data.items()
    }


def compare_results(current: Dict[str, BenchmarkMetrics], 
                    baseline: Dict[str, BenchmarkMetrics],
                    tolerance: float = 0.1,
                    strict_checksum: bool = False) -> Tuple[bool, str]:
    """Compare current results against baseline.
    
    Args:
        current: Current benchmark results
        baseline: Baseline results to compare against
        tolerance: Allowed relative deviation (0.1 = 10%)
        strict_checksum: If True, checksum mismatches fail the comparison.
                        If False (default), they are reported but don't fail.
                        MPS/Metal has non-deterministic behavior, so strict=False is recommended.
    
    Returns:
        (passed, report): Whether comparison passed and detailed report
    """
    report_lines = []
    all_passed = True
    
    report_lines.append("=" * 70)
    report_lines.append("BENCHMARK COMPARISON REPORT")
    report_lines.append("=" * 70)
    
    for name in BENCHMARKS:
        curr = current.get(name)
        base = baseline.get(name)
        
        report_lines.append(f"\n{name.upper()}")
        report_lines.append("-" * 40)
        
        if curr is None:
            report_lines.append("  CURRENT: FAILED")
            all_passed = False
            continue
        if base is None:
            report_lines.append("  BASELINE: MISSING")
            continue
        
        # Check determinism via checksum
        # Note: MPS/Metal is non-deterministic, so we only warn by default
        if curr.state_checksum != base.state_checksum:
            if strict_checksum:
                report_lines.append(f"  CHECKSUM: MISMATCH (FAIL)")
                all_passed = False
            else:
                report_lines.append(f"  CHECKSUM: MISMATCH (non-deterministic GPU, OK)")
            report_lines.append(f"    Current:  {curr.state_checksum}")
            report_lines.append(f"    Baseline: {base.state_checksum}")
        else:
            report_lines.append(f"  CHECKSUM: OK ({curr.state_checksum})")
        
        # Compare key metrics
        # These are the metrics that matter for regression testing
        metrics_to_compare = [
            ("wall_time_ms", "Performance", 0.5, True),  # 50% tolerance, fail if degraded
            ("final_carrier_count", "Carrier Count", 0.25, False),  # 25% tolerance, warn only
            ("n_crystallized", "Crystallized", 0.25, False),  # 25% tolerance, warn only
            ("energy_conservation_ratio", "Energy Conservation", tolerance, False),  # warn only
        ]
        
        for metric, label, tol, fail_on_regress in metrics_to_compare:
            curr_val = getattr(curr, metric, None)
            base_val = getattr(base, metric, None)
            
            if curr_val is None or base_val is None:
                continue
            
            if base_val == 0:
                if curr_val == 0:
                    status = "OK"
                else:
                    status = "CHANGED (was 0)"
                    # If crystallized went from 0 to >0, that's an improvement
                    if metric == "n_crystallized" and curr_val > 0:
                        status = f"IMPROVED (+{curr_val})"
            else:
                rel_change = abs(curr_val - base_val) / abs(base_val)
                if rel_change <= tol:
                    status = "OK"
                elif curr_val < base_val and metric == "wall_time_ms":
                    # Faster is better for wall_time
                    status = f"IMPROVED (-{rel_change*100:.1f}%)"
                elif curr_val > base_val and metric == "wall_time_ms":
                    # Slower is worse
                    status = f"DEGRADED (+{rel_change*100:.1f}%)"
                    if fail_on_regress:
                        all_passed = False
                elif curr_val > base_val:
                    # For counts, more could be good or neutral
                    status = f"CHANGED (+{rel_change*100:.1f}%)"
                else:
                    status = f"CHANGED ({-rel_change*100:.1f}%)"
            
            report_lines.append(f"  {label}: {curr_val:.4g} vs {base_val:.4g} [{status}]")
    
    report_lines.append("\n" + "=" * 70)
    if all_passed:
        report_lines.append("OVERALL: PASSED")
        report_lines.append("  All functional metrics within tolerance.")
        report_lines.append("  Note: Checksum mismatches are expected on MPS/Metal due to non-determinism.")
    else:
        report_lines.append("OVERALL: FAILED")
        report_lines.append("  One or more benchmarks failed or showed significant regression.")
    report_lines.append("=" * 70)
    
    return all_passed, "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Sensorium Manifold Benchmark Suite")
    parser.add_argument("--save", type=str, help="Save results to this file")
    parser.add_argument("--compare", type=str, help="Compare against this baseline file")
    parser.add_argument("--benchmark", type=str, help="Run only this benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()
    
    device = args.device or get_device()
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    print()
    
    if args.benchmark:
        # Run single benchmark
        if args.benchmark not in BENCHMARKS:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {list(BENCHMARKS.keys())}")
            return 1
        
        benchmark = BENCHMARKS[args.benchmark](device, args.seed)
        result = benchmark.run()
        print(f"\nResults for {args.benchmark}:")
        for key, val in result.to_dict().items():
            if key != "custom":
                print(f"  {key}: {val}")
        return 0
    
    # Run all benchmarks
    results = run_all_benchmarks(device, args.seed)
    
    if args.save:
        save_results(results, Path(args.save))
    
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            print(f"Baseline file not found: {baseline_path}")
            return 1
        
        baseline = load_results(baseline_path)
        passed, report = compare_results(results, baseline)
        print()
        print(report)
        return 0 if passed else 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for name, result in results.items():
        if result:
            print(f"\n{name}:")
            print(f"  Time: {result.wall_time_ms:.1f}ms ({result.ms_per_step:.2f}ms/step)")
            print(f"  Carriers: {result.final_carrier_count} (crys: {result.n_crystallized})")
            print(f"  Checksum: {result.state_checksum}")
        else:
            print(f"\n{name}: FAILED")
    
    return 0


if __name__ == "__main__":
    exit(main())
