"""Count aggregate observer.

Counts the number of items in an observation.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Count:
    """Count the number of items in an observation.
    
    This aggregate returns the count of items.
    
    Example:
        observer = Count()(Crystallized()(Modes()))
        count = observer.observe(state)  # Returns int
        
        # Or use fluent style:
        count = Modes().observe(state).count()
    """
    
    def __call__(self, inner: ObserverProtocol) -> _CountObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that counts the inner observer's result
        """
        return _CountObserver(inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> int:
        """Count items directly from an ObservationResult.
        
        Args:
            state: ObservationResult to count
            **kwargs: Ignored
        
        Returns:
            Number of items
        """
        if not isinstance(state, ObservationResult):
            return 0
        return len(state.items)


class _CountObserver:
    """Internal observer that wraps another observer with counting."""
    
    def __init__(self, inner: ObserverProtocol):
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> int:
        """Observe with the inner observer, then count."""
        result = self.inner.observe(state, **kwargs)
        return len(result.items)
