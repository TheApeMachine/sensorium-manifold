"""Oscillators primitive observer.

Extracts oscillator state from the simulation, including phase,
frequency (omega), and energy. Automatically filters dark particles.
"""

from __future__ import annotations

from typing import Any

from sensorium.observers.types import (
    ObservationResult,
    PARTICLE_FLAG_DARK,
    to_numpy,
    get_visible_mask,
)


class Oscillators:
    """Extract oscillator state from the simulation.
    
    Returns an ObservationResult with items representing each visible oscillator.
    Dark particles are automatically filtered out.
    
    Each item contains: index, phase, omega, energy.
    
    Example:
        result = Oscillators().observe(state)
        high_freq = result.where(lambda o: o["omega"] > 0.5)
        freq_stats = result.statistics("omega")
    """
    
    def __init__(self, include_dark: bool = False):
        """Initialize the oscillators observer.
        
        Args:
            include_dark: If True, include oscillators from dark particles
        """
        self.include_dark = include_dark
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Extract oscillator data from state.
        
        Args:
            state: Simulation state dict or previous ObservationResult
            **kwargs: Additional context
        
        Returns:
            ObservationResult with oscillator items
        """
        # Handle ObservationResult input
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return ObservationResult(data={"items": []}, source="oscillators")
        
        # Extract arrays
        osc_phase = raw_state.get("osc_phase")
        osc_omega = raw_state.get("osc_omega") or raw_state.get("excitations")
        osc_energy = raw_state.get("osc_energy")
        particle_flags = raw_state.get("particle_flags")
        
        # Need at least phase or omega
        if osc_phase is None and osc_omega is None:
            return ObservationResult(data={"items": []}, source="oscillators")
        
        # Determine number of oscillators
        if osc_phase is not None:
            n = len(to_numpy(osc_phase))
        else:
            n = len(to_numpy(osc_omega))
        
        # Get visibility mask
        if self.include_dark:
            visible_mask = [True] * n
        else:
            visible_mask = get_visible_mask(particle_flags, n)
        
        visible_indices = [i for i in range(n) if visible_mask[i]]
        
        # Convert to numpy
        phase_np = to_numpy(osc_phase) if osc_phase is not None else None
        omega_np = to_numpy(osc_omega) if osc_omega is not None else None
        energy_np = to_numpy(osc_energy) if osc_energy is not None else None
        
        # Build items list
        items = []
        for idx in visible_indices:
            item = {"index": int(idx)}
            
            if phase_np is not None:
                item["phase"] = float(phase_np[idx])
            
            if omega_np is not None:
                item["omega"] = float(omega_np[idx])
                item["frequency"] = float(omega_np[idx])  # Alias
            
            if energy_np is not None:
                item["energy"] = float(energy_np[idx])
            
            items.append(item)
        
        # Summary statistics
        data = {
            "items": items,
            "n_visible": len(items),
            "n_total": n,
        }
        
        # Add frequency statistics if available
        if omega_np is not None:
            visible_omega = omega_np[visible_mask]
            if len(visible_omega) > 0:
                data["omega_min"] = float(visible_omega.min())
                data["omega_max"] = float(visible_omega.max())
                data["omega_mean"] = float(visible_omega.mean())
                data["omega_std"] = float(visible_omega.std())
        
        return ObservationResult(data=data, source="oscillators", _items=items)
