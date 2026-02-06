"""Energy metric observers.

Measures energy accumulation and correlation with collisions.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObservationResult, to_numpy
from sensorium.observers.primitives import Tokens, Particles


class EnergyAccumulation:
    """Measure total energy accumulated in colliding tokens.
    
    Tokens with more collisions should accumulate more energy
    (information density). This computes the correlation between
    collision count and total energy.
    
    Example:
        corr = EnergyAccumulation().observe(state)
        # Positive correlation = energy accumulates with collisions
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute energy-collision correlation."""
        tokens_result = Tokens().observe(state)
        
        if not tokens_result or len(tokens_result.items) < 2:
            return 0.0
        
        # Extract counts and energies from token items
        counts = [item.get("count", 0) for item in tokens_result.items]
        energies = [item.get("total_energy", 0) for item in tokens_result.items]
        
        if len(counts) < 2:
            return 0.0
        
        # Compute Pearson correlation
        corr = np.corrcoef(counts, energies)[0, 1]
        
        if np.isnan(corr):
            return 0.0
        
        return float(corr)


class EnergyCorrelation:
    """Compute energy-collision correlation with baseline comparison.
    
    Returns both actual correlation and a shuffled baseline to
    measure excess correlation.
    
    Example:
        result = EnergyCorrelation().observe(state)
        # Returns dict: {actual, baseline, excess}
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> dict:
        """Compute energy correlation with baseline."""
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return {"actual": 0.0, "baseline": 0.0, "excess": 0.0}
        
        energies = raw_state.get("energies")
        token_ids = raw_state.get("token_ids")
        
        if energies is None or token_ids is None:
            return {"actual": 0.0, "baseline": 0.0, "excess": 0.0}
        
        energies_np = to_numpy(energies)
        token_ids_np = to_numpy(token_ids)
        
        # Compute actual correlation
        actual = self._compute_correlation(token_ids_np, energies_np)
        
        # Compute baseline with shuffled token IDs
        token_ids_shuffled = token_ids_np.copy()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(token_ids_shuffled)
        baseline = self._compute_correlation(token_ids_shuffled, energies_np)
        
        return {
            "actual": actual,
            "baseline": baseline,
            "excess": actual - baseline,
        }
    
    def _compute_correlation(
        self,
        token_ids: np.ndarray,
        energies: np.ndarray,
    ) -> float:
        """Compute correlation between collision count and total energy."""
        # Aggregate by token ID
        token_energy_map = {}
        token_count_map = {}
        
        for token_id, energy in zip(token_ids, energies):
            tid = int(token_id)
            if tid not in token_energy_map:
                token_energy_map[tid] = 0.0
                token_count_map[tid] = 0
            token_energy_map[tid] += float(energy)
            token_count_map[tid] += 1
        
        if len(token_count_map) < 2:
            return 0.0
        
        counts = list(token_count_map.values())
        energies = list(token_energy_map.values())
        
        corr = np.corrcoef(counts, energies)[0, 1]
        
        if np.isnan(corr):
            return 0.0
        
        return float(corr)


class MeanParticleEnergy:
    """Compute mean particle energy.
    
    Example:
        mean_e = MeanParticleEnergy().observe(state)
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute mean particle energy."""
        particles_result = Particles().observe(state)
        
        if not particles_result:
            return 0.0
        
        return particles_result.mean("energy")
