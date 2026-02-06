"""Mean aggregate observer.

Computes the mean of a field across all items.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Mean:
    """Compute the mean of a field across all items.
    
    This aggregate returns the average of a specified field.
    
    Example:
        observer = Mean("energy")(Particles())
        mean_energy = observer.observe(state)  # Returns float
        
        # Or use fluent style:
        mean_energy = Particles().observe(state).mean("energy")
    """
    
    def __init__(self, field: str):
        """Initialize the mean aggregate.
        
        Args:
            field: The field name to average
        """
        self.field = field
    
    def __call__(self, inner: ObserverProtocol) -> _MeanObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that averages the field from the inner observer's result
        """
        return _MeanObserver(self.field, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Compute mean directly from an ObservationResult.
        
        Args:
            state: ObservationResult to aggregate
            **kwargs: Ignored
        
        Returns:
            Mean of field values
        """
        if not isinstance(state, ObservationResult):
            return 0.0
        
        values = [item.get(self.field, 0) for item in state.items]
        if not values:
            return 0.0
        return sum(values) / len(values)


class _MeanObserver:
    """Internal observer that wraps another observer with averaging."""
    
    def __init__(self, field: str, inner: ObserverProtocol):
        self.field = field
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Observe with the inner observer, then average."""
        result = self.inner.observe(state, **kwargs)
        
        values = [item.get(self.field, 0) for item in result.items]
        if not values:
            return 0.0
        return sum(values) / len(values)
