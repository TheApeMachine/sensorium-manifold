"""Image-based collision experiment: proving hash collisions = compression with real images.

Clean pattern:
- Datasets: SyntheticDataset with COLLISION pattern (image-sized units)
- Observers: Composable metric observers
- Projectors: LaTeX tables and figures

Writes:
- `paper/tables/image_collision_summary.tex`
- `paper/figures/image_collision_hero.png`
- `paper/figures/image_collision_bifurcation.png`
"""

from __future__ import annotations

from pathlib import Path

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, GeometricSimulationConfig, CoherenceSimulationConfig
from optimizer.tokenizer import TokenizerConfig

# =============================================================================
# DATASET
# =============================================================================

from sensorium.dataset import SyntheticDataset, SyntheticConfig, SyntheticPattern

# MNIST image constants
IMAGE_SIZE = 28 * 28  # 784 bytes per image

# =============================================================================
# OBSERVERS
# =============================================================================

from sensorium.observers import observe_reaction, Particles, Modes
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    SpatialClustering,
    ClusteringExcess,
    CompressionRatio,
    CollisionRate,
    TokenEntropy,
    EnergyCorrelation,
    MeanParticleEnergy,
)

# =============================================================================
# PROJECTORS
# =============================================================================

from sensorium.projectors import (
    PipelineProjector,
    LaTeXTableProjector,
    FigureProjector,
)


# =============================================================================
# EXPERIMENT
# =============================================================================

class ImageCollisionExperiment(Experiment):
    """Demonstrate hash collisions = compression using image-sized data.
    
    Uses SyntheticDataset with COLLISION pattern and 784-byte units
    (matching MNIST image size).
    
    Clean pattern:
    - datasets: List of SyntheticDatasets with different collision rates
    - manifold: Runs simulation
    - inference: Composes observers for metrics  
    - projector: Outputs tables and figures
    """

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Collision rates to test
        self.collision_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.num_images = 50
        
        # Create datasets for each collision rate
        self.datasets = [
            SyntheticDataset(SyntheticConfig(
                pattern=SyntheticPattern.COLLISION,
                num_units=self.num_images,
                unit_length=IMAGE_SIZE,
                collision_rate=rate,
                seed=42,
            ))
            for rate in self.collision_rates
        ]
        
        # Manifold configuration
        self.manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                tokenizer=TokenizerConfig(
                    hash_vocab_size=4096,
                    hash_prime=31,
                ),
                geometric=GeometricSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                    poisson_iterations=50,
                ),
                coherence=CoherenceSimulationConfig(
                    grid_size=(32, 32, 32),
                    dt=0.01,
                    poisson_iterations=50,
                ),
                position_init="random",
                position_init_seed=42,
            ),
            observers={
                "geometric": observe_reaction(Particles()),
                "coherence": observe_reaction(Modes()),
            }
        )
        
        # Inference observer composes metric observers
        self.inference = InferenceObserver([
            SpatialClustering(),
            ClusteringExcess(),
            CompressionRatio(),
            CollisionRate(),
            TokenEntropy(),
            EnergyCorrelation(),
            MeanParticleEnergy(),
        ])
        
        # Projector pipeline for outputs
        self.projector = PipelineProjector(
            LaTeXTableProjector(output_dir=self.artifact_path("tables")),
            FigureProjector(output_dir=self.artifact_path("figures")),
        )
        
        # Store results across datasets
        self.results = []

    def run(self):
        """Run experiment for each collision rate."""
        for i, dataset in enumerate(self.datasets):
            collision_rate = self.collision_rates[i]
            
            print(f"\n{'='*60}")
            print(f"Testing collision rate: {collision_rate} (image size: {IMAGE_SIZE} bytes)")
            print('='*60)
            
            # Add dataset and run
            self.manifold.add_dataset(dataset.generate)
            state = self.manifold.run()
            
            # Observe metrics
            observation = self.observe(state)
            observation["collision_rate"] = collision_rate
            observation["num_images"] = self.num_images
            observation["image_size"] = IMAGE_SIZE
            
            # Log to console
            self._log_metrics(observation)
            
            self.results.append(observation)
        
        # Project final results
        self.project(self._prepare_projection())
        
        # Generate additional visualizations
        self._create_hero_visualization()
        self._create_bifurcation_charts()
    
    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        result = self.inference.observe(state)
        # Keep raw state for visualization
        result["_state"] = state
        return result
    
    def project(self, observation: dict) -> dict:
        """Project observation results to artifacts."""
        return self.projector.project(observation)
    
    def _prepare_projection(self) -> dict:
        """Prepare data for projection from all results."""
        # Table data
        table_data = [
            {
                "Rate": r["collision_rate"],
                "Compression": r.get("compression_ratio", 0),
                "Clustering": r.get("spatial_clustering", 0),
                "Entropy": r.get("entropy", 0),
                "Images": r.get("num_images", 0),
            }
            for r in self.results
        ]
        
        # Figure data
        figure_data = {
            "Compression": [r.get("compression_ratio", 0) for r in self.results],
            "Clustering": [r.get("spatial_clustering", 0) for r in self.results],
            "Entropy": [r.get("entropy", 0) for r in self.results],
        }
        
        return {
            # Table projection
            "table_name": "image_collision_summary",
            "table_caption": "Image collision compression metrics (784 bytes/image)",
            "table_columns": ["Rate", "Compression", "Clustering", "Entropy", "Images"],
            "table_data": table_data,
            
            # Figure projection
            "figure_name": "image_collision_metrics",
            "figure_type": "line",
            "figure_data": figure_data,
            "figure_title": "Image Collision Rate vs Compression Metrics",
            "figure_xlabel": "Collision Rate",
            "figure_ylabel": "Metric Value",
            "figure_grid": True,
        }
    
    def _log_metrics(self, metrics: dict):
        """Log metrics to console."""
        print(f"\n  Compression Metrics:")
        print(f"    Compression ratio: {metrics.get('compression_ratio', 0):.4f}")
        print(f"    Collision rate:    {metrics.get('collision_rate', 0):.3f}")
        print(f"    Entropy:           {metrics.get('entropy', 0):.2f} bits")
        
        print(f"\n  Clustering Metrics:")
        print(f"    Spatial clustering: {metrics.get('spatial_clustering', 0):.4f}")
        print(f"    Clustering excess:  {metrics.get('clustering_excess', 0):.4f}")
    
    def _create_hero_visualization(self):
        """Create hero visualization for the paper."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        
        if len(self.results) < 3:
            return
        
        # Select representative results (low, medium, high collision rates)
        low_result = self.results[0]  # 0.1
        mid_result = self.results[len(self.results) // 2]  # 0.5
        high_result = self.results[-1]  # 0.9
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25)
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        label_idx = 0
        
        # Row 1: 3D spatial clustering for different collision rates
        for idx, result in enumerate([low_result, mid_result, high_result]):
            state = result.get("_state", {})
            positions = state.get("positions")
            token_ids = state.get("token_ids")
            energies = state.get("energies")
            
            if positions is None or token_ids is None:
                continue
            
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().numpy()
            if isinstance(energies, torch.Tensor):
                energies = energies.detach().cpu().numpy()
            
            ax = fig.add_subplot(gs[0, idx], projection='3d')
            
            unique_tokens = np.unique(token_ids)
            token_to_color = {tid: i / len(unique_tokens) for i, tid in enumerate(unique_tokens)}
            colors = np.array([token_to_color[tid] for tid in token_ids])
            
            sizes = (energies - energies.min()) / (energies.max() - energies.min() + 1e-10) * 80 + 10
            
            ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes, cmap='viridis', alpha=0.6, edgecolors='none'
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=45)
            ax.text2D(0.02, 0.98, labels[label_idx], transform=ax.transAxes, 
                     fontsize=14, fontweight='bold', va='top')
            label_idx += 1
        
        # Row 2: Energy accumulation scatter plots
        for idx, result in enumerate([low_result, mid_result, high_result]):
            state = result.get("_state", {})
            token_ids = state.get("token_ids")
            energies = state.get("energies")
            
            if token_ids is None or energies is None:
                continue
            
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().numpy()
            if isinstance(energies, torch.Tensor):
                energies = energies.detach().cpu().numpy()
            
            ax = fig.add_subplot(gs[1, idx])
            
            token_counts = {}
            token_energies = {}
            for tid, energy in zip(token_ids, energies):
                if tid not in token_counts:
                    token_counts[tid] = 0
                    token_energies[tid] = []
                token_counts[tid] += 1
                token_energies[tid].append(energy)
            
            counts = list(token_counts.values())
            total_energies = [np.sum(token_energies[tid]) for tid in token_counts.keys()]
            
            ax.scatter(counts, total_energies, alpha=0.6, s=40, c='#F18F01', 
                      edgecolors='black', linewidth=0.3)
            ax.set_xlabel('Particles per token')
            ax.set_ylabel('Total energy')
            
            if len(counts) > 1:
                z = np.polyfit(counts, total_energies, 1)
                p = np.poly1d(z)
                ax.plot(sorted(counts), p(sorted(counts)), "r--", alpha=0.8, linewidth=2)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, labels[label_idx], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top')
            label_idx += 1
        
        # Row 3: 2D projections and compression curve
        for idx, result in enumerate([low_result, mid_result]):
            state = result.get("_state", {})
            positions = state.get("positions")
            token_ids = state.get("token_ids")
            
            if positions is None or token_ids is None:
                continue
            
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().numpy()
            
            ax = fig.add_subplot(gs[2, idx])
            
            unique_tokens = np.unique(token_ids)
            token_to_color = {tid: i / len(unique_tokens) for i, tid in enumerate(unique_tokens)}
            colors = np.array([token_to_color[tid] for tid in token_ids])
            
            ax.scatter(positions[:, 0], positions[:, 1], c=colors, 
                      cmap='viridis', alpha=0.6, s=15, edgecolors='none')
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, labels[label_idx], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top')
            label_idx += 1
        
        # Panel I: Compression metrics curve
        ax = fig.add_subplot(gs[2, 2])
        collision_rates = [r["collision_rate"] for r in self.results]
        compression_ratios = [r.get("compression_ratio", 0) for r in self.results]
        entropies = [r.get("entropy", 0) for r in self.results]
        
        ax2 = ax.twinx()
        ax.plot(collision_rates, compression_ratios, 'o-', linewidth=2, markersize=8, 
               color='#2E86AB', label='Compression')
        ax2.plot(collision_rates, entropies, 's-', linewidth=2, markersize=8,
                color='#A23B72', label='Entropy')
        
        ax.set_xlabel('Collision rate')
        ax.set_ylabel('Compression ratio', color='#2E86AB')
        ax2.set_ylabel('Entropy (bits)', color='#A23B72')
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.text(0.02, 0.98, labels[label_idx], transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        path = self.artifact_path("figures", "image_collision_hero.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Generated: {path}")
    
    def _create_bifurcation_charts(self):
        """Create bifurcation charts showing trie structure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial import ConvexHull
        
        if len(self.results) < 3:
            return
        
        low_result = self.results[0]
        mid_result = self.results[len(self.results) // 2]
        high_result = self.results[-1]
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        labels_top = ['A', 'B', 'C']
        labels_bottom = ['D', 'E', 'F']
        
        for idx, result in enumerate([low_result, mid_result, high_result]):
            state = result.get("_state", {})
            positions = state.get("positions")
            token_ids = state.get("token_ids")
            
            if positions is None or token_ids is None:
                continue
            
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().numpy()
            
            # Row 1: Hierarchical clustering dendrogram
            ax1 = fig.add_subplot(gs[0, idx])
            
            unique_tokens = np.unique(token_ids)
            if len(unique_tokens) > 100:
                sample_indices = np.linspace(0, len(unique_tokens)-1, 50, dtype=int)
                sample_tokens = unique_tokens[sample_indices]
            else:
                sample_tokens = unique_tokens
            
            token_positions = {}
            for tid in sample_tokens:
                mask = token_ids == tid
                if np.any(mask):
                    token_positions[tid] = positions[mask].mean(axis=0)
            
            if len(token_positions) > 1:
                pos_array = np.array(list(token_positions.values()))
                linkage_matrix = linkage(pos_array, method='ward')
                dendrogram(linkage_matrix, ax=ax1, leaf_rotation=90, leaf_font_size=5,
                          no_labels=True, color_threshold=0)
                ax1.set_xlabel('Token clusters')
                ax1.set_ylabel('Distance')
            
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.02, 0.98, labels_top[idx], transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', va='top')
            
            # Row 2: Spatial bifurcation
            ax2 = fig.add_subplot(gs[1, idx])
            
            if len(unique_tokens) > 20:
                token_to_group = {tid: i % 20 for i, tid in enumerate(unique_tokens)}
                colors = np.array([token_to_group[tid] for tid in token_ids])
            else:
                token_to_color = {tid: i / len(unique_tokens) for i, tid in enumerate(unique_tokens)}
                colors = np.array([token_to_color[tid] for tid in token_ids])
            
            ax2.scatter(positions[:, 0], positions[:, 1], c=colors, 
                       cmap='tab20', alpha=0.6, s=12, edgecolors='none')
            
            # Draw convex hulls for first few token groups
            for tid in unique_tokens[:10]:
                mask = token_ids == tid
                if np.sum(mask) > 2:
                    same_token_pos = positions[mask]
                    try:
                        hull = ConvexHull(same_token_pos[:, :2])
                        for simplex in hull.simplices:
                            ax2.plot(same_token_pos[simplex, 0], same_token_pos[simplex, 1], 
                                    'k-', alpha=0.15, linewidth=0.5)
                    except:
                        pass
            
            ax2.set_xlabel('X position')
            ax2.set_ylabel('Y position')
            ax2.set_aspect('equal')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.02, 0.98, labels_bottom[idx], transform=ax2.transAxes, 
                    fontsize=14, fontweight='bold', va='top')
        
        plt.tight_layout()
        path = self.artifact_path("figures", "image_collision_bifurcation.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Generated: {path}")
