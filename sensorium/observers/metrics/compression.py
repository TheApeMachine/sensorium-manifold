"""Compression metric observers.

Measures compression ratio and collision statistics.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult
from sensorium.observers.primitives import Tokens


class CompressionRatio:
    """Compute compression ratio: unique_tokens / total_particles.
    
    Lower ratio = more compression (more collisions).
    - 1.0 = no compression (every particle has unique token)
    - 0.0 = maximum compression (all particles same token)
    
    Example:
        ratio = CompressionRatio().observe(state)
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute compression ratio."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result:
            return 1.0
        
        n_unique = tokens_result.get("n_unique", 0)
        n_particles = tokens_result.get("n_particles", 0)
        
        if n_particles == 0:
            return 1.0
        
        return float(n_unique / n_particles)


class CollisionCount:
    """Count total number of collisions (particles sharing token IDs).
    
    A collision occurs when multiple particles have the same token ID.
    This counts total_particles - unique_tokens.
    
    Example:
        collisions = CollisionCount().observe(state)
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> int:
        """Count collisions."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result:
            return 0
        
        n_unique = tokens_result.get("n_unique", 0)
        n_particles = tokens_result.get("n_particles", 0)
        
        return max(0, n_particles - n_unique)


class CollisionRate:
    """Compute collision rate: 1 - (unique / total).
    
    Higher rate = more collisions = more compression.
    
    Example:
        rate = CollisionRate().observe(state)
        # 0.0 = no collisions
        # 0.9 = 90% of particles share tokens
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute collision rate."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result:
            return 0.0
        
        # collision_ratio is already computed by Tokens observer
        return float(tokens_result.get("collision_ratio", 0.0))


class CollidingTokens:
    """Get tokens that have collisions (count > 1).
    
    Returns an ObservationResult with only tokens that have
    multiple particles.
    
    Example:
        result = CollidingTokens().observe(state)
        n_colliding = result.count()
        top_colliders = result.top_k(5, by="count")
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Get colliding tokens."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result:
            return ObservationResult(data={"items": []}, source="colliding_tokens")
        
        # Filter to tokens with count > 1
        return tokens_result.where(lambda t: t.get("count", 0) > 1)
