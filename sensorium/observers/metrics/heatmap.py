"""Heatmap/matrix metric observers.

Observers that produce 2D matrices for visualization.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObservationResult, to_numpy


class CollisionMatrix:
    """Generate a collision matrix: token_id vs position.
    
    This creates a 2D histogram showing which token IDs appear
    at which positions, useful for visualizing collision patterns.
    
    Example:
        matrix = CollisionMatrix(bins=64, segment_size=32).observe(state)
        # Returns 2D numpy array for heatmap plotting
    """
    
    def __init__(self, bins: int = 64, segment_size: int = 32):
        """Initialize collision matrix observer.
        
        Args:
            bins: Number of bins for token ID axis
            segment_size: Segment size for position wrapping
        """
        self.bins = bins
        self.segment_size = segment_size
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate collision matrix.
        
        Args:
            state: Simulation state with token_ids
            
        Returns:
            2D numpy array of shape (bins, segment_size)
        """
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return np.zeros((self.bins, self.segment_size))
        
        token_ids = raw_state.get("token_ids")
        
        if token_ids is None:
            return np.zeros((self.bins, self.segment_size))
        
        token_ids_np = to_numpy(token_ids)
        n = len(token_ids_np)
        
        if n == 0:
            return np.zeros((self.bins, self.segment_size))
        
        # Position within segment (wraps at segment_size)
        pos_1d = (np.arange(n) % self.segment_size).astype(np.int32)
        
        # Normalize token IDs to [0, bins-1]
        tid_min = token_ids_np.min()
        tid_max = token_ids_np.max()
        tid_range = tid_max - tid_min + 1e-10
        token_ids_normalized = (
            (token_ids_np - tid_min) / tid_range * (self.bins - 1)
        ).astype(int)
        
        # Build matrix
        matrix = np.zeros((self.bins, self.segment_size))
        for pos, tid in zip(pos_1d, token_ids_normalized):
            pos_bin = min(int(pos), self.segment_size - 1)
            tid_bin = min(max(int(tid), 0), self.bins - 1)
            matrix[tid_bin, pos_bin] += 1
        
        return matrix


class TokenPositionDensity:
    """Compute density of tokens at each position.
    
    Useful for seeing which positions have more collisions.
    
    Example:
        density = TokenPositionDensity(segment_size=32).observe(state)
        # Returns 1D array of length segment_size
    """
    
    def __init__(self, segment_size: int = 32):
        self.segment_size = segment_size
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute position density."""
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return np.zeros(self.segment_size)
        
        token_ids = raw_state.get("token_ids")
        
        if token_ids is None:
            return np.zeros(self.segment_size)
        
        n = len(to_numpy(token_ids))
        
        if n == 0:
            return np.zeros(self.segment_size)
        
        # Count unique tokens at each position
        positions = np.arange(n) % self.segment_size
        density = np.zeros(self.segment_size)
        
        for pos in range(self.segment_size):
            mask = positions == pos
            unique_at_pos = len(np.unique(to_numpy(token_ids)[mask]))
            total_at_pos = np.sum(mask)
            if total_at_pos > 0:
                # Collision ratio at this position
                density[pos] = 1.0 - (unique_at_pos / total_at_pos)
        
        return density
