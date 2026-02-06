"""Where transform observer.

Filters observation items based on a predicate function.
"""

from __future__ import annotations

from typing import Callable

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Where:
    """Filter observation items based on a predicate.
    
    This transform keeps only items that satisfy the predicate.
    
    Example:
        observer = Where(lambda m: m["amplitude"] > 0.5)(Modes())
        result = observer.observe(state)
        
        # Or use directly:
        result = Where(lambda m: m["state"] == 2).observe(mode_result)
    """
    
    def __init__(self, predicate: Callable[[dict], bool]):
        """Initialize the where transform.
        
        Args:
            predicate: Function that takes an item dict and returns True to keep
        """
        self.predicate = predicate
    
    def __call__(self, inner: ObserverProtocol) -> _WhereObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that filters the inner observer's result
        """
        return _WhereObserver(self.predicate, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Apply filter directly to an ObservationResult.
        
        Args:
            state: ObservationResult to filter
            **kwargs: Ignored
        
        Returns:
            ObservationResult with filtered items
        """
        if not isinstance(state, ObservationResult):
            return ObservationResult(data={"items": []}, source="where")
        
        items = [item for item in state.items if self.predicate(item)]
        
        return ObservationResult(
            data={**state.data, "items": items},
            source=state.source,
            _items=items,
        )


class _WhereObserver:
    """Internal observer that wraps another observer with filtering."""
    
    def __init__(self, predicate: Callable[[dict], bool], inner: ObserverProtocol):
        self.predicate = predicate
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe with the inner observer, then filter."""
        result = self.inner.observe(state, **kwargs)
        
        items = [item for item in result.items if self.predicate(item)]
        
        return ObservationResult(
            data={**result.data, "items": items},
            source=result.source,
            _items=items,
        )
