"""Count metric observers.

Simple observers that count particles, modes, and other entities.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult


class ParticleCount:
    """Count total number of particles in the simulation.
    
    Example:
        count = ParticleCount().observe(state)
        # Returns: {"n_particles": 1000}
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> dict:
        """Count particles."""
        if state is None:
            return {"n_particles": 0}
        
        # Handle dict or ObservationResult
        data = state.data if hasattr(state, "data") else state
        
        # Try different sources for particle count
        token_ids = data.get("token_ids")
        if token_ids is not None:
            if hasattr(token_ids, "__len__"):
                return {"n_particles": len(token_ids)}
            elif hasattr(token_ids, "numel"):
                return {"n_particles": int(token_ids.numel())}
        
        positions = data.get("positions")
        if positions is not None:
            if hasattr(positions, "shape"):
                return {"n_particles": int(positions.shape[0])}
            elif hasattr(positions, "__len__"):
                return {"n_particles": len(positions)}
        
        return {"n_particles": 0}


class ModeCount:
    """Count total number of active modes in the simulation.
    
    A mode is considered active if its amplitude > threshold.
    
    Example:
        count = ModeCount().observe(state)
        # Returns: {"n_modes": 42}
    """
    
    def __init__(self, threshold: float = 1e-6):
        """
        Args:
            threshold: Minimum amplitude to count as active mode
        """
        self.threshold = threshold
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        *,
        manifold=None,
        **kwargs,
    ) -> dict:
        """Count active modes."""
        if state is None:
            return {"n_modes": 0}

        data = state.data if hasattr(state, "data") else state
        amplitudes = data.get("amplitudes")
        if amplitudes is None:
            return {"n_modes": 0}
        
        # Count modes with amplitude above threshold
        if hasattr(amplitudes, "sum"):  # torch tensor
            n_modes = int((amplitudes > self.threshold).sum().item())
        else:
            import numpy as np
            amplitudes = np.array(amplitudes)
            n_modes = int((amplitudes > self.threshold).sum())
        
        return {"n_modes": n_modes}


class CrystallizedCount:
    """Count number of crystallized modes.
    
    Crystallized modes have mode_state == 2.
    
    Example:
        count = CrystallizedCount().observe(state, manifold=manifold)
        # Returns: {"n_crystallized": 10}
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        *,
        manifold=None,
        **kwargs,
    ) -> dict:
        """Count crystallized modes."""
        if state is None:
            return {"n_crystallized": 0}

        data = state.data if hasattr(state, "data") else state
        mode_state = data.get("mode_state")
        if mode_state is None:
            return {"n_crystallized": 0}

        # Count modes with state == 2 (crystallized)
        if hasattr(mode_state, "sum"):  # torch tensor
            n_crystallized = int((mode_state == 2).sum().item())
        else:
            import numpy as np
            mode_state = np.array(mode_state)
            n_crystallized = int((mode_state == 2).sum())
        
        return {"n_crystallized": n_crystallized}


class UniqueTokenCount:
    """Count number of unique token IDs.
    
    Example:
        count = UniqueTokenCount().observe(state)
        # Returns: {"n_unique_tokens": 256}
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> dict:
        """Count unique tokens."""
        if state is None:
            return {"n_unique_tokens": 0}
        
        data = state.data if hasattr(state, "data") else state
        token_ids = data.get("token_ids")
        
        if token_ids is None:
            return {"n_unique_tokens": 0}
        
        if hasattr(token_ids, "unique"):  # torch tensor
            n_unique = int(token_ids.unique().numel())
        else:
            import numpy as np
            token_ids = np.array(token_ids)
            n_unique = len(np.unique(token_ids))
        
        return {"n_unique_tokens": n_unique}
