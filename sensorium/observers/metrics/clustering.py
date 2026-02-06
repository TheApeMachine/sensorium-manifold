"""Spatial clustering metric observers.

Measures how well particles with the same token ID cluster together
in physical space - a key indicator of compression.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObservationResult, to_numpy
from sensorium.observers.primitives import Tokens


class SpatialClustering:
    """Compute spatial clustering score for particles by token ID.
    
    Particles with the same token ID should cluster together if
    compression is working. This observer computes the inverse of
    the mean intra-token distance.
    
    Higher score = tighter clustering = more compression.
    
    Example:
        score = SpatialClustering().observe(state)
        # Returns float: clustering score
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute spatial clustering score.
        
        Args:
            state: Simulation state with positions and token_ids
            
        Returns:
            Clustering score (higher = tighter clustering)
        """
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return 0.0
        
        positions = raw_state.get("positions")
        token_ids = raw_state.get("token_ids")
        
        if positions is None or token_ids is None:
            return 0.0
        
        positions_np = to_numpy(positions)
        token_ids_np = to_numpy(token_ids)
        
        return self._compute_clustering(positions_np, token_ids_np)
    
    def _compute_clustering(
        self,
        positions: np.ndarray,
        token_ids: np.ndarray,
    ) -> float:
        """Compute clustering score from arrays."""
        clustering_distances = []
        
        for token_id in np.unique(token_ids):
            same_token_mask = token_ids == token_id
            if np.sum(same_token_mask) < 2:
                continue
            
            same_token_positions = positions[same_token_mask]
            if len(same_token_positions) > 1:
                # Compute pairwise distances
                distances = []
                for i in range(len(same_token_positions)):
                    for j in range(i + 1, len(same_token_positions)):
                        dist = np.linalg.norm(
                            same_token_positions[i] - same_token_positions[j]
                        )
                        distances.append(dist)
                if distances:
                    clustering_distances.append(np.mean(distances))
        
        if not clustering_distances:
            return 0.0
        
        # Inverse of mean distance = clustering score
        return float(1.0 / (1.0 + np.mean(clustering_distances)))


class ClusteringBaseline:
    """Compute baseline clustering score with shuffled token IDs.
    
    This provides a null hypothesis baseline - what clustering would
    we see by random chance? The difference between actual and baseline
    is the "excess clustering" due to compression.
    
    Example:
        actual = SpatialClustering().observe(state)
        baseline = ClusteringBaseline().observe(state)
        excess = actual - baseline  # Compression signal
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute baseline clustering with shuffled tokens.
        
        Args:
            state: Simulation state with positions and token_ids
            
        Returns:
            Baseline clustering score
        """
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return 0.0
        
        positions = raw_state.get("positions")
        token_ids = raw_state.get("token_ids")
        
        if positions is None or token_ids is None:
            return 0.0
        
        positions_np = to_numpy(positions)
        token_ids_np = to_numpy(token_ids).copy()
        
        # Shuffle token IDs
        rng = np.random.RandomState(self.seed)
        rng.shuffle(token_ids_np)
        
        # Use SpatialClustering logic
        return SpatialClustering()._compute_clustering(positions_np, token_ids_np)


class ClusteringExcess:
    """Compute excess clustering: actual - baseline.
    
    This is the key metric that shows compression is happening
    above random chance.
    
    Example:
        excess = ClusteringExcess().observe(state)
        # Positive = real clustering, negative = anti-clustering
    """
    
    def __init__(self, seed: int = 42):
        self.clustering = SpatialClustering()
        self.baseline = ClusteringBaseline(seed=seed)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute excess clustering score."""
        actual = self.clustering.observe(state, **kwargs)
        baseline = self.baseline.observe(state, **kwargs)
        return actual - baseline
