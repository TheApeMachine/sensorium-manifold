"""Tokens primitive observer.

Extracts token state from the simulation, including token IDs,
positions, and associated energies. Automatically filters dark particles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sensorium.observers.types import (
    ObservationResult,
    PARTICLE_FLAG_DARK,
    to_numpy,
    get_visible_mask,
)


class Tokens:
    """Extract token state from the simulation.
    
    Returns an ObservationResult with items representing unique tokens.
    Dark particles are automatically filtered out.
    
    Each item contains: token_id, count, total_energy, mean_energy, positions.
    
    Example:
        result = Tokens().observe(state)
        top_tokens = result.top_k(10, by="total_energy")
        collision_rate = 1.0 - (result.count() / result.get("n_particles"))
    """
    
    def __init__(self, include_dark: bool = False):
        """Initialize the tokens observer.
        
        Args:
            include_dark: If True, include tokens from dark particles
        """
        self.include_dark = include_dark
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Extract token data from state.
        
        Args:
            state: Simulation state dict or previous ObservationResult
            **kwargs: Additional context
        
        Returns:
            ObservationResult with token items
        """
        # Handle ObservationResult input
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return ObservationResult(data={"items": []}, source="tokens")
        
        # Extract arrays
        token_ids = raw_state.get("token_ids")
        energies = raw_state.get("energies")
        positions = raw_state.get("positions")
        particle_flags = raw_state.get("particle_flags")
        
        # No tokens if token_ids missing
        if token_ids is None:
            return ObservationResult(data={"items": []}, source="tokens")
        
        # Convert to numpy
        token_ids_np = to_numpy(token_ids)
        n = len(token_ids_np)
        
        # Get visibility mask
        if self.include_dark:
            visible_mask = [True] * n
        else:
            visible_mask = get_visible_mask(particle_flags, n)
        
        # Aggregate by token_id
        token_data = defaultdict(lambda: {
            "count": 0,
            "total_energy": 0.0,
            "positions": [],
            "indices": [],
        })
        
        energies_np = to_numpy(energies) if energies is not None else None
        positions_np = to_numpy(positions) if positions is not None else None
        
        for i in range(n):
            if not visible_mask[i]:
                continue
            
            tid = int(token_ids_np[i])
            token_data[tid]["count"] += 1
            token_data[tid]["indices"].append(i)
            
            if energies_np is not None:
                token_data[tid]["total_energy"] += float(energies_np[i])
            
            if positions_np is not None:
                token_data[tid]["positions"].append(positions_np[i].tolist())
        
        # Build items list
        items = []
        for tid, info in token_data.items():
            item = {
                "token_id": tid,
                "count": info["count"],
                "total_energy": info["total_energy"],
                "mean_energy": info["total_energy"] / info["count"] if info["count"] > 0 else 0.0,
                "indices": info["indices"],
            }
            
            if info["positions"]:
                # Compute centroid
                import numpy as np
                pos_arr = np.array(info["positions"])
                item["centroid"] = pos_arr.mean(axis=0).tolist()
                
                # Compute spatial spread (std of distances from centroid)
                if len(pos_arr) > 1:
                    centroid = pos_arr.mean(axis=0)
                    distances = np.linalg.norm(pos_arr - centroid, axis=1)
                    item["spatial_spread"] = float(distances.std())
                else:
                    item["spatial_spread"] = 0.0
            
            items.append(item)
        
        # Summary statistics
        n_visible = sum(1 for m in visible_mask if m)
        n_unique = len(items)
        
        data = {
            "items": items,
            "n_unique": n_unique,
            "n_particles": n_visible,
            "collision_ratio": 1.0 - (n_unique / n_visible) if n_visible > 0 else 0.0,
        }
        
        return ObservationResult(data=data, source="tokens", _items=items)
