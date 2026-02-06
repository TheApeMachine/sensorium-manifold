"""Statistics aggregate observer.

Computes comprehensive statistics for a field across all items.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Statistics:
    """Compute comprehensive statistics for a field.
    
    This aggregate returns min, max, mean, std, and percentiles
    for the specified field.
    
    Example:
        observer = Statistics("omega")(Oscillators())
        stats = observer.observe(state)
        # Returns: {"count": N, "min": ..., "max": ..., "mean": ..., ...}
        
        # Or use fluent style:
        stats = Oscillators().observe(state).statistics("omega")
    """
    
    def __init__(self, field: str, percentiles: list[int] | None = None):
        """Initialize the statistics aggregate.
        
        Args:
            field: The field name to compute statistics for
            percentiles: List of percentiles to compute (default: [10, 50, 90])
        """
        self.field = field
        self.percentiles = percentiles or [10, 50, 90]
    
    def __call__(self, inner: ObserverProtocol) -> _StatisticsObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that computes statistics from the inner observer's result
        """
        return _StatisticsObserver(self.field, self.percentiles, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute statistics directly from an ObservationResult.
        
        Args:
            state: ObservationResult to aggregate
            **kwargs: Ignored
        
        Returns:
            Dictionary with statistical measures
        """
        if not isinstance(state, ObservationResult):
            return self._empty_stats()
        
        values = [item.get(self.field, 0) for item in state.items]
        return self._compute_stats(values)
    
    def _compute_stats(self, values: list[float]) -> dict[str, float]:
        """Compute statistics from a list of values."""
        if not values:
            return self._empty_stats()
        
        arr = np.array(values)
        
        result = {
            "count": len(values),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "sum": float(arr.sum()),
        }
        
        # Add percentiles
        for p in self.percentiles:
            result[f"p{p}"] = float(np.percentile(arr, p))
        
        return result
    
    def _empty_stats(self) -> dict[str, float]:
        """Return empty statistics dict."""
        result = {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "sum": 0.0,
        }
        for p in self.percentiles:
            result[f"p{p}"] = 0.0
        return result


class _StatisticsObserver:
    """Internal observer that wraps another observer with statistics."""
    
    def __init__(
        self,
        field: str,
        percentiles: list[int],
        inner: ObserverProtocol,
    ):
        self.field = field
        self.percentiles = percentiles
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Observe with the inner observer, then compute statistics."""
        result = self.inner.observe(state, **kwargs)
        
        values = [item.get(self.field, 0) for item in result.items]
        
        if not values:
            return self._empty_stats()
        
        arr = np.array(values)
        
        stats = {
            "count": len(values),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "sum": float(arr.sum()),
        }
        
        for p in self.percentiles:
            stats[f"p{p}"] = float(np.percentile(arr, p))
        
        return stats
    
    def _empty_stats(self) -> dict[str, float]:
        """Return empty statistics dict."""
        result = {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "sum": 0.0,
        }
        for p in self.percentiles:
            result[f"p{p}"] = 0.0
        return result
