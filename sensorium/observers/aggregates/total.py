"""Total aggregate observer.

Sums the values of a field across all items.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Total:
    """Sum the values of a field across all items.
    
    This aggregate returns the sum of a specified field.
    
    Example:
        observer = Total("amplitude")(Modes())
        total_amp = observer.observe(state)  # Returns float
        
        # Or use fluent style:
        total_amp = Modes().observe(state).total("amplitude")
    """
    
    def __init__(self, field: str):
        """Initialize the total aggregate.
        
        Args:
            field: The field name to sum
        """
        self.field = field
    
    def __call__(self, inner: ObserverProtocol) -> _TotalObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that sums the field from the inner observer's result
        """
        return _TotalObserver(self.field, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Sum field directly from an ObservationResult.
        
        Args:
            state: ObservationResult to aggregate
            **kwargs: Ignored
        
        Returns:
            Sum of field values
        """
        if not isinstance(state, ObservationResult):
            return 0.0
        return sum(item.get(self.field, 0) for item in state.items)


class _TotalObserver:
    """Internal observer that wraps another observer with summing."""
    
    def __init__(self, field: str, inner: ObserverProtocol):
        self.field = field
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> float:
        """Observe with the inner observer, then sum."""
        result = self.inner.observe(state, **kwargs)
        return sum(item.get(self.field, 0) for item in result.items)
