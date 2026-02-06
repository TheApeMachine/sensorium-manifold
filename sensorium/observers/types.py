"""Core types for the composable observer system.

This module defines the foundation types for building chainable, composable
observations of the simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, TypeVar, Union
import numpy as np
import torch


# =============================================================================
# Particle Flags
# =============================================================================

PARTICLE_FLAG_DARK = 1 << 0  # Dark particle: non-coupling, observer-invisible


# =============================================================================
# Mode State Constants
# =============================================================================

MODE_VOLATILE = 0
MODE_STABLE = 1
MODE_CRYSTALLIZED = 2


# =============================================================================
# ObservationResult
# =============================================================================

@dataclass
class ObservationResult:
    """Typed result that enables fluent method chaining.
    
    This is the core data structure returned by all observers. It wraps
    the observation data and provides fluent methods for filtering,
    transforming, and aggregating results.
    
    Example:
        result = Modes().observe(state)
        crystallized_count = result.where(
            lambda m: m["state"] == MODE_CRYSTALLIZED
        ).count()
    """
    
    data: dict = field(default_factory=dict)
    source: str = "unknown"  # e.g., "carriers", "particles", "tokens"
    _items: list[dict] | None = field(default=None, repr=False)
    
    def __post_init__(self):
        """Extract items from data if available."""
        if self._items is None and "items" in self.data:
            self._items = self.data["items"]
    
    @property
    def items(self) -> list[dict]:
        """Get the list of items in this observation."""
        if self._items is not None:
            return self._items
        if "items" in self.data:
            return self.data["items"]
        return []
    
    # -------------------------------------------------------------------------
    # Fluent filtering methods
    # -------------------------------------------------------------------------
    
    def take(self, n: int) -> ObservationResult:
        """Take the first n items."""
        items = self.items[:n]
        return ObservationResult(
            data={**self.data, "items": items},
            source=self.source,
            _items=items,
        )
    
    def where(self, predicate: Callable[[dict], bool]) -> ObservationResult:
        """Filter items based on a predicate."""
        items = [item for item in self.items if predicate(item)]
        return ObservationResult(
            data={**self.data, "items": items},
            source=self.source,
            _items=items,
        )
    
    def sort_by(self, key: str, descending: bool = True) -> ObservationResult:
        """Sort items by a field."""
        items = sorted(
            self.items,
            key=lambda x: x.get(key, 0),
            reverse=descending,
        )
        return ObservationResult(
            data={**self.data, "items": items},
            source=self.source,
            _items=items,
        )
    
    def top_k(self, k: int, by: str) -> ObservationResult:
        """Get top-k items by a field."""
        return self.sort_by(by, descending=True).take(k)
    
    def bottom_k(self, k: int, by: str) -> ObservationResult:
        """Get bottom-k items by a field."""
        return self.sort_by(by, descending=False).take(k)
    
    # -------------------------------------------------------------------------
    # Aggregation methods
    # -------------------------------------------------------------------------
    
    def count(self) -> int:
        """Count the number of items."""
        return len(self.items)
    
    def total(self, field: str | None = None) -> float:
        """Sum values of a field, or count if no field specified."""
        if field is None:
            return float(len(self.items))
        return sum(item.get(field, 0) for item in self.items)
    
    def mean(self, field: str) -> float:
        """Compute the mean of a field."""
        values = [item.get(field, 0) for item in self.items]
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def min(self, field: str) -> float:
        """Get the minimum value of a field."""
        values = [item.get(field, 0) for item in self.items]
        if not values:
            return 0.0
        return min(values)
    
    def max(self, field: str) -> float:
        """Get the maximum value of a field."""
        values = [item.get(field, 0) for item in self.items]
        if not values:
            return 0.0
        return max(values)
    
    def std(self, field: str) -> float:
        """Compute the standard deviation of a field."""
        values = [item.get(field, 0) for item in self.items]
        if len(values) < 2:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    def statistics(self, field: str) -> dict[str, float]:
        """Compute comprehensive statistics for a field."""
        values = [item.get(field, 0) for item in self.items]
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "std": 0}
        
        n = len(values)
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n if n > 1 else 0
        
        sorted_values = sorted(values)
        
        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": mean_val,
            "std": variance ** 0.5,
            "p10": sorted_values[int(n * 0.1)] if n >= 10 else sorted_values[0],
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)] if n >= 10 else sorted_values[-1],
        }
    
    # -------------------------------------------------------------------------
    # Data access methods
    # -------------------------------------------------------------------------
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dict."""
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to data."""
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in data."""
        return key in self.data
    
    def __len__(self) -> int:
        """Return the number of items."""
        return len(self.items)
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over items."""
        return iter(self.items)
    
    def __bool__(self) -> bool:
        """Truth value based on whether there are items."""
        return len(self.items) > 0
    
    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return self.data.copy()
    
    def values(self, field: str) -> list[Any]:
        """Get all values for a field across items."""
        return [item.get(field) for item in self.items]
    
    def as_tensor(self, field: str, device: str = "cpu") -> torch.Tensor:
        """Get field values as a tensor."""
        values = self.values(field)
        return torch.tensor(values, device=device)
    
    def as_array(self, field: str) -> np.ndarray:
        """Get field values as a numpy array."""
        return np.array(self.values(field))


# =============================================================================
# Observer Protocol
# =============================================================================

class ObserverProtocol(Protocol):
    """Protocol for all observers in the system.
    
    Observers receive state (as ObservationResult or dict) and return
    an ObservationResult. This enables chaining observers together.
    """
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe the state and return a result.
        
        Args:
            state: Input state to observe. Can be:
                - ObservationResult from a previous observer
                - dict containing raw simulation state
                - None for observers that don't need input
            **kwargs: Additional context (e.g., manifold reference)
        
        Returns:
            ObservationResult with observed data
        """
        ...


# =============================================================================
# Helper functions
# =============================================================================

def to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def get_visible_mask(
    particle_flags: torch.Tensor | np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Get mask of visible (non-dark) particles.
    
    Args:
        particle_flags: Tensor of particle flags, or None
        n: Number of particles
    
    Returns:
        Boolean mask where True = visible particle
    """
    if particle_flags is None:
        return np.ones(n, dtype=bool)
    
    flags = to_numpy(particle_flags)
    return (flags & PARTICLE_FLAG_DARK) == 0
