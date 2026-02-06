"""Particles primitive observer.

Extracts particle state from the simulation, including positions,
velocities, energies, and heats. Automatically filters dark particles.
"""

from __future__ import annotations

from typing import Any

from sensorium.observers.types import (
    ObservationResult,
    PARTICLE_FLAG_DARK,
    to_numpy,
    get_visible_mask,
)


class Particles:
    """Extract particle state from the simulation.
    
    Returns an ObservationResult with items representing each visible particle.
    Dark particles are automatically filtered out.
    
    Each item contains: index, position, velocity, energy, heat, mass.
    
    Example:
        result = Particles().observe(state)
        high_energy = result.where(lambda p: p["energy"] > 1.0)
        hottest = result.top_k(10, by="heat")
    """
    
    def __init__(self, include_dark: bool = False):
        """Initialize the particles observer.
        
        Args:
            include_dark: If True, include dark particles in results
        """
        self.include_dark = include_dark
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Extract particle data from state.
        
        Args:
            state: Simulation state dict or previous ObservationResult
            **kwargs: Additional context
        
        Returns:
            ObservationResult with particle items
        """
        # Handle ObservationResult input
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return ObservationResult(data={"items": []}, source="particles")
        
        # Extract particle arrays
        positions = raw_state.get("positions")
        velocities = raw_state.get("velocities")
        energies = raw_state.get("energies")
        heats = raw_state.get("heats")
        masses = raw_state.get("masses")
        particle_flags = raw_state.get("particle_flags")
        
        # No particles if positions missing
        if positions is None:
            return ObservationResult(data={"items": []}, source="particles")
        
        # Convert to numpy
        positions_np = to_numpy(positions)
        n = len(positions_np)
        
        # Get visibility mask (filters dark particles)
        if self.include_dark:
            visible_mask = [True] * n
        else:
            visible_mask = get_visible_mask(particle_flags, n)
        
        visible_indices = [i for i in range(n) if visible_mask[i]]
        
        # Build items list
        items = []
        for idx in visible_indices:
            item = {
                "index": int(idx),
                "position": positions_np[idx].tolist(),
                "x": float(positions_np[idx, 0]),
                "y": float(positions_np[idx, 1]),
                "z": float(positions_np[idx, 2]),
            }
            
            if velocities is not None:
                vel = to_numpy(velocities)[idx]
                item["velocity"] = vel.tolist()
                item["speed"] = float((vel ** 2).sum() ** 0.5)
            
            if energies is not None:
                item["energy"] = float(to_numpy(energies)[idx])
            
            if heats is not None:
                item["heat"] = float(to_numpy(heats)[idx])
            
            if masses is not None:
                item["mass"] = float(to_numpy(masses)[idx])
            
            items.append(item)
        
        # Summary statistics
        data = {
            "items": items,
            "n_visible": len(items),
            "n_total": n,
        }
        
        if not self.include_dark and particle_flags is not None:
            flags_np = to_numpy(particle_flags)
            data["n_dark"] = int(((flags_np & PARTICLE_FLAG_DARK) != 0).sum())
        
        return ObservationResult(data=data, source="particles", _items=items)
