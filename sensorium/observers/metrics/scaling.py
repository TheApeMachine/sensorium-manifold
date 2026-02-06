"""Scaling experiment observer for tracking mode dynamics.

Tracks mode population, births, deaths, and conflict over time
for scaling analysis experiments.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

from sensorium.observers.types import ObserverProtocol


class ModeTrackingObserver(ObserverProtocol):
    """Observer that tracks mode dynamics over time."""
    
    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.step_count = 0
        
        # Time series data
        self.history = {
            "step": [],
            "n_modes": [],
            "n_volatile": [],
            "n_stable": [],
            "n_crystallized": [],
            "max_amplitude": [],
            "mean_amplitude": [],
            "n_births": [],  # New modes this step
            "n_deaths": [],  # Pruned modes this step
            "conflict_score": [],
        }
        self._prev_mode_count = 0
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        self.step_count += 1
        
        if observation is None:
            return {"done_thinking": self.step_count >= self.max_steps}
        
        data = observation.data if hasattr(observation, "data") else observation
        
        # Extract mode info
        amplitudes = data.get("amplitudes")
        mode_state = data.get("mode_state")
        conflict = data.get("conflict")
        
        if amplitudes is not None:
            active_mask = amplitudes > 1e-6
            n_modes = int(active_mask.sum().item())
            max_amp = float(amplitudes.max().item()) if n_modes > 0 else 0.0
            mean_amp = float(amplitudes[active_mask].mean().item()) if n_modes > 0 else 0.0
        else:
            n_modes = 0
            max_amp = 0.0
            mean_amp = 0.0
        
        # Count states: 0=volatile, 1=stable, 2=crystallized
        n_volatile = n_stable = n_crystallized = 0
        if mode_state is not None and n_modes > 0:
            states = mode_state[:n_modes]
            n_volatile = int((states == 0).sum().item())
            n_stable = int((states == 1).sum().item())
            n_crystallized = int((states == 2).sum().item())
        
        # Births and deaths
        n_births = max(0, n_modes - self._prev_mode_count)
        n_deaths = max(0, self._prev_mode_count - n_modes)
        self._prev_mode_count = n_modes
        
        # Conflict score
        conflict_score = 0.0
        if conflict is not None and n_modes > 0:
            conflict_score = float(conflict[:n_modes].mean().item())
        
        # Record
        self.history["step"].append(self.step_count)
        self.history["n_modes"].append(n_modes)
        self.history["n_volatile"].append(n_volatile)
        self.history["n_stable"].append(n_stable)
        self.history["n_crystallized"].append(n_crystallized)
        self.history["max_amplitude"].append(max_amp)
        self.history["mean_amplitude"].append(mean_amp)
        self.history["n_births"].append(n_births)
        self.history["n_deaths"].append(n_deaths)
        self.history["conflict_score"].append(conflict_score)
        
        return {"done_thinking": self.step_count >= self.max_steps}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from tracking history."""
        if not self.history["n_modes"]:
            return {}
        
        total_births = sum(self.history["n_births"])
        total_deaths = sum(self.history["n_deaths"])
        
        return {
            "n_modes_final": self.history["n_modes"][-1] if self.history["n_modes"] else 0,
            "n_crystallized_final": self.history["n_crystallized"][-1] if self.history["n_crystallized"] else 0,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "pruning_rate": total_deaths / (total_births + 1),
            "steps": self.step_count,
        }
