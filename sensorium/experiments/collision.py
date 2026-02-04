from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import torch
import numpy as np

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, GeometricSimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig, Tokenizer
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.state import StateObserver
from sensorium.observers.carrier import CarrierObserver


class CollisionDataset:
    """Dataset designed to create controlled hash collisions for compression demonstration.
    
    Creates multiple "files" where:
    - Same byte sequences at same positions → same token ID (collision = compression)
    - Different sequences → different token IDs (bifurcation = trie structure)
    """
    
    def __init__(
        self,
        num_files: int = 20,
        file_length: int = 32,
        collision_rate: float = 0.5,  # Fraction of positions that collide across files
        seed: int = 42,
    ):
        self.num_files = num_files
        self.file_length = file_length
        self.collision_rate = collision_rate
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Generate base pattern (shared across files at collision positions)
        self.base_pattern = self._rng.randint(0, 256, size=file_length, dtype=np.uint8)
        
        # Generate unique patterns for non-collision positions
        self.unique_patterns = [
            self._rng.randint(0, 256, size=file_length, dtype=np.uint8)
            for _ in range(num_files)
        ]
        
        # Determine which positions collide (same across files)
        num_collisions = int(file_length * collision_rate)
        self.collision_positions = set(
            self._rng.choice(file_length, size=num_collisions, replace=False)
        )
    
    def generate(self) -> Iterator[bytes]:
        """Generate files with controlled collisions.
        
        IMPORTANT: For collisions to occur, same bytes must appear at same positions
        across files. The tokenizer uses: token_id = (byte * prime + position) & mask
        With segment_size set, position wraps, so same (byte, position) → same token_id.
        """
        for file_idx in range(self.num_files):
            file_bytes = bytearray(self.file_length)
            
            for pos in range(self.file_length):
                if pos in self.collision_positions:
                    # Collision: same byte across all files at this position
                    # This ensures: hash(byte, pos) is the same for all files
                    file_bytes[pos] = self.base_pattern[pos]
                else:
                    # Unique: different byte per file at this position
                    # This ensures: hash(byte, pos) differs across files
                    file_bytes[pos] = self.unique_patterns[file_idx][pos]
            
            yield bytes(file_bytes)


class CollisionExperiment(Experiment):
    """Demonstrate that hash collisions act as compression in a thermodynamic trie.
    
    EXPERIMENT DESIGN:
    ==================
    This experiment proves that hash collisions in the Sensorium Manifold act as
    compression, similar to a thermodynamic trie structure.
    
    Key Hypothesis:
    - When multiple inputs map to the same token ID (hash collision), they should
      behave as if they're the same entity, leading to spatial clustering and
      energy accumulation (compression).
    - This creates a trie-like structure where:
      * Shared prefixes (collisions) → particles cluster together
      * Divergences (unique tokens) → particles separate (bifurcation)
    
    Controlled Variables:
    - Collision rate: Fraction of positions that collide across files (0.1 to 0.9)
    - Creates multiple "files" where some positions share bytes (collisions)
      and others are unique (bifurcations)
    
    Metrics to Prove Compression:
    1. Spatial Clustering: Particles with same token ID should cluster together
       - Lower inter-particle distance for same token ID = more compression
    2. Energy Accumulation: Colliding particles accumulate more energy
       - Multiple inputs → same particle → higher energy (information density)
    3. Entropy Reduction: System entropy decreases as collisions increase
       - Fewer unique states = more compression (Shannon entropy)
    4. Compression Ratio: unique_tokens / total_particles
       - Lower ratio = more collisions = more compression
    
    Expected Results:
    - As collision_rate increases:
      ✓ Spatial clustering increases (particles cluster by token ID)
      ✓ Compression ratio decreases (fewer unique tokens)
      ✓ Entropy decreases (information compression)
      ✓ Energy accumulates in colliding particles
    
    This demonstrates that the system naturally compresses redundant information
    through thermodynamic dynamics, creating a trie-like hierarchical structure.
    """

    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        # Test different collision rates to show compression scaling
        self.collision_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Also test different vocab sizes to show collision vs aliasing regimes
        self.vocab_sizes = [256, 512, 1024, 2048, 4096]

    def _compute_compression_metrics(self, state: dict, tokenizer: Tokenizer = None) -> dict:
        """Compute metrics that prove compression from hash collisions."""
        positions = state.get("positions")
        energies = state.get("energies")
        token_ids = state.get("token_ids")
        
        if positions is None or token_ids is None:
            return {
                "spatial_clustering": 0.0,
                "energy_accumulation": 0.0,
                "entropy": 0.0,
                "unique_tokens": 0,
                "total_particles": 0,
            }
        
        # Convert to numpy
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy()
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        
        unique_tokens = len(np.unique(token_ids))
        total_particles = len(token_ids)

        # 1. Spatial clustering score, plus a shuffled baseline to avoid tautologies.
        def _clustering_score(ids: np.ndarray) -> float:
            clustering_distances = []
            for token_id in np.unique(ids):
                same_token_mask = ids == token_id
                if np.sum(same_token_mask) < 2:
                    continue

                same_token_positions = positions[same_token_mask]
                if len(same_token_positions) > 1:
                    distances = []
                    for i in range(len(same_token_positions)):
                        for j in range(i + 1, len(same_token_positions)):
                            dist = np.linalg.norm(same_token_positions[i] - same_token_positions[j])
                            distances.append(dist)
                    if distances:
                        clustering_distances.append(np.mean(distances))
            return float(1.0 / (1.0 + np.mean(clustering_distances))) if clustering_distances else 0.0

        spatial_clustering = _clustering_score(token_ids)
        token_ids_shuf = token_ids.copy()
        np.random.shuffle(token_ids_shuf)
        spatial_clustering_baseline = _clustering_score(token_ids_shuf)
        spatial_clustering_excess = spatial_clustering - spatial_clustering_baseline

        # 2. Energy accumulation correlation, plus shuffled baseline.
        def _energy_corr(ids: np.ndarray) -> float:
            token_energy_map = {}
            token_count_map = {}
            for token_id, energy in zip(ids, energies):
                if token_id not in token_energy_map:
                    token_energy_map[token_id] = []
                    token_count_map[token_id] = 0
                token_energy_map[token_id].append(energy)
                token_count_map[token_id] += 1

            total_energies = [np.sum(es) for es in token_energy_map.values()]
            collision_counts = list(token_count_map.values())
            if total_energies and collision_counts and len(collision_counts) > 1:
                c = np.corrcoef(collision_counts, total_energies)[0, 1]
                return float(0.0 if np.isnan(c) else c)
            return 0.0

        energy_accumulation = _energy_corr(token_ids)
        token_ids_shuf2 = token_ids.copy()
        np.random.shuffle(token_ids_shuf2)
        energy_accumulation_baseline = _energy_corr(token_ids_shuf2)
        energy_accumulation_excess = energy_accumulation - energy_accumulation_baseline
        
        # 3. Entropy: Lower entropy = more compression (fewer unique states)
        # Shannon entropy of token distribution
        token_counts = {}
        for token_id in token_ids:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        probs = np.array(list(token_counts.values())) / total_particles
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Compression ratio: unique tokens / total particles
        # Lower ratio = more compression (more collisions)
        compression_ratio = unique_tokens / total_particles if total_particles > 0 else 1.0
        
        return {
            "spatial_clustering": float(spatial_clustering),
            "spatial_clustering_baseline": float(spatial_clustering_baseline),
            "spatial_clustering_excess": float(spatial_clustering_excess),
            "energy_accumulation": float(energy_accumulation),
            "energy_accumulation_baseline": float(energy_accumulation_baseline),
            "energy_accumulation_excess": float(energy_accumulation_excess),
            "entropy": float(entropy),
            "compression_ratio": float(compression_ratio),
            "unique_tokens": int(unique_tokens),
            "total_particles": int(total_particles),
        }
    
    def observe(self, state: dict, metrics: dict = None):
        """Visualize compression metrics and collision patterns."""
        positions = state.get("positions")
        token_ids = state.get("token_ids")
        
        if positions is None or token_ids is None:
            return
        
        # Convert to numpy
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        
        # Create collision matrix: token_id vs byte position (within segment).
        # This avoids relying on the geometric init policy for interpretation.
        segment_size = int(metrics.get("segment_size", 0)) if metrics else 0
        if segment_size <= 0:
            segment_size = 32
        bins = 64
        pos_1d = (np.arange(len(token_ids)) % segment_size).astype(np.int32)
        
        # Normalize token IDs to [0, bins-1] for visualization
        token_ids_normalized = ((token_ids - token_ids.min()) / 
                                (token_ids.max() - token_ids.min() + 1e-10) * (bins - 1)).astype(int)
        
        collision_matrix = np.zeros((bins, segment_size))
        for pos, tid in zip(pos_1d, token_ids_normalized):
            pos_bin = min(int(pos), segment_size - 1)
            tid_bin = min(int(tid), bins - 1)
            collision_matrix[tid_bin, pos_bin] += 1
        
        # Add metrics as text overlay
        title = "Hash Collision = Compression (Thermodynamic Trie)"
        if metrics:
            title += f"\nClustering(excess): {metrics.get('spatial_clustering_excess', 0.0):.3f}, "
            title += f"Compression: {metrics['compression_ratio']:.3f}, "
            title += f"Entropy: {metrics['entropy']:.2f}"
        
        self.plot_heatmap(
            matrix=collision_matrix,
            name="collision_compression",
            title=title,
            xlabel="Byte position (within segment)",
            ylabel="Token ID (normalized)",
            cmap="viridis",
            figsize=(12, 10),
            fmt="png",
        )

    def run(self):
        """Run collision experiment to prove hash collisions = compression."""
        results = []
        
        # Test different collision rates
        for collision_rate in self.collision_rates:
            dataset = CollisionDataset(
                num_files=20,
                file_length=32,
                collision_rate=collision_rate,
                seed=42,
            )
            
            # Create tokenizer config with segment_size to reset index per file
            # This ensures that same bytes at same positions across files create collisions
            tokenizer_config = TokenizerConfig(
                hash_vocab_size=4096,
                hash_prime=31,
                segment_size=dataset.file_length,  # Reset index after each file
            )
            
            # Run simulation (token_ids will be included in state)
            manifold = Manifold(
                SimulationConfig(
                    dashboard=False,
                    generator=dataset.generate,
                    geometric=GeometricSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                        poisson_iterations=50,
                    ),
                    spectral=SpectralSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                        poisson_iterations=50,
                    ),
                    tokenizer=tokenizer_config,
                    # Integrity: do NOT seed positions from token_id in this experiment.
                    position_init="random",
                    position_init_seed=42,
                ),
                observers={
                    "geometric": InferenceObserver([StateObserver()]),
                    "spectral": InferenceObserver([CarrierObserver(None)])
                }
            )
            
            state = manifold.run()
            
            # Create tokenizer for metrics computation (just for interface)
            temp_tokenizer = Tokenizer(tokenizer_config)
            
            # Compute compression metrics
            metrics = self._compute_compression_metrics(state, temp_tokenizer)
            metrics["collision_rate"] = collision_rate
            metrics["segment_size"] = dataset.file_length
            
            # Debug: Verify collisions are actually happening
            token_ids = state.get("token_ids")
            if token_ids is not None:
                if isinstance(token_ids, torch.Tensor):
                    token_ids_np = token_ids.detach().cpu().numpy()
                else:
                    token_ids_np = token_ids
                unique_tokens = len(np.unique(token_ids_np))
                total_particles = len(token_ids_np)
                actual_collision_rate = 1.0 - (unique_tokens / total_particles) if total_particles > 0 else 0.0
                metrics["actual_collision_rate"] = actual_collision_rate
                
                # Expected unique tokens if collisions work perfectly:
                # collision_positions × 1 + unique_positions × num_files
                expected_unique = (
                    len(dataset.collision_positions) * 1 +
                    (dataset.file_length - len(dataset.collision_positions)) * dataset.num_files
                )
                metrics["expected_unique_tokens"] = expected_unique
                metrics["collision_efficiency"] = (
                    (expected_unique - unique_tokens) / (expected_unique - dataset.file_length)
                    if expected_unique > dataset.file_length else 0.0
                )
                
                print(f"Collision rate {collision_rate:.1f}: "
                      f"unique={unique_tokens}, total={total_particles}, "
                      f"actual_collision_rate={actual_collision_rate:.3f}, "
                      f"expected_unique={expected_unique}, "
                      f"efficiency={metrics['collision_efficiency']:.3f}")
            
            results.append((state, metrics))
            
            # Visualize
            self.observe(state, metrics)
        
        # Create summary plot showing compression vs collision rate
        self._plot_compression_summary(results)
    
    def _plot_compression_summary(self, results):
        """Plot summary showing compression metrics vs collision rate."""
        import matplotlib.pyplot as plt
        
        collision_rates = [r[1]["collision_rate"] for r in results]
        clustering = [r[1]["spatial_clustering"] for r in results]
        compression_ratios = [r[1]["compression_ratio"] for r in results]
        entropies = [r[1]["entropy"] for r in results]
        actual_collision_rates = [r[1].get("actual_collision_rate", 0.0) for r in results]
        
        path = self.artifact_path("figures", "collision_compression_summary.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top row: Expected metrics
        # Spatial clustering increases with collision rate (compression)
        axes[0, 0].plot(collision_rates, clustering, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Target Collision Rate")
        axes[0, 0].set_ylabel("Spatial Clustering")
        axes[0, 0].set_title("Spatial Clustering\n(Higher = Particles Cluster by Token ID)")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Compression ratio: should DECREASE with collision rate
        # (fewer unique tokens relative to total = more compression)
        axes[0, 1].plot(collision_rates, compression_ratios, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel("Target Collision Rate")
        axes[0, 1].set_ylabel("Compression Ratio (unique/total)")
        axes[0, 1].set_title("Compression Ratio\n(Lower = More Compression)")
        axes[0, 1].grid(True, alpha=0.3)
        # Add reference line showing ideal compression
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='No compression')
        axes[0, 1].legend()
        
        # Bottom row: Actual collision rate and entropy
        # Show actual vs target collision rate
        axes[1, 0].plot(collision_rates, actual_collision_rates, 'o-', linewidth=2, markersize=8, color='purple', label='Actual')
        axes[1, 0].plot(collision_rates, collision_rates, '--', linewidth=1, color='gray', alpha=0.5, label='Target')
        axes[1, 0].set_xlabel("Target Collision Rate")
        axes[1, 0].set_ylabel("Actual Collision Rate")
        axes[1, 0].set_title("Collision Rate Verification\n(Should Match Target)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Entropy: should DECREASE with collision rate (information compression)
        axes[1, 1].plot(collision_rates, entropies, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel("Target Collision Rate")
        axes[1, 1].set_ylabel("Shannon Entropy (bits)")
        axes[1, 1].set_title("Information Entropy\n(Lower = More Compression)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
