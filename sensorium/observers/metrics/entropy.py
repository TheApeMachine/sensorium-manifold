"""Entropy metric observers.

Measures information entropy of the token distribution -
lower entropy means more compression (fewer unique states).
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObservationResult, to_numpy
from sensorium.observers.primitives import Tokens


class TokenEntropy:
    """Compute Shannon entropy of the token distribution.
    
    Lower entropy = more compression (fewer unique states, more collisions).
    
    Example:
        entropy = TokenEntropy().observe(state)
        # Returns float: entropy in bits
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute token entropy.
        
        Args:
            state: Simulation state with token_ids
            
        Returns:
            Shannon entropy in bits
        """
        # Use Tokens observer to get token data
        tokens_result = Tokens().observe(state)
        
        if not tokens_result or not tokens_result.items:
            return 0.0
        
        # Get counts from token items
        counts = [item.get("count", 0) for item in tokens_result.items]
        total = sum(counts)
        
        if total == 0:
            return 0.0
        
        # Compute probabilities
        probs = np.array(counts) / total
        
        # Shannon entropy: -sum(p * log2(p))
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)


class NormalizedEntropy:
    """Compute normalized entropy (0 = max compression, 1 = no compression).
    
    Normalized by maximum possible entropy (uniform distribution).
    
    Example:
        norm_entropy = NormalizedEntropy().observe(state)
        # 0.0 = all particles same token (max compression)
        # 1.0 = all particles different tokens (no compression)
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute normalized entropy."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result:
            return 1.0
        
        n_unique = tokens_result.get("n_unique", 0)
        n_particles = tokens_result.get("n_particles", 0)
        
        if n_unique <= 1 or n_particles <= 1:
            return 0.0
        
        actual_entropy = TokenEntropy().observe(state, **kwargs)
        max_entropy = np.log2(n_unique)  # Uniform distribution
        
        if max_entropy == 0:
            return 0.0
        
        return float(actual_entropy / max_entropy)
