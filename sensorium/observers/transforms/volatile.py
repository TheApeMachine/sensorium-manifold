"""Volatile transform observer.

Filters to only volatile modes.
"""

from __future__ import annotations

from sensorium.observers.types import (
    ObservationResult,
    ObserverProtocol,
    MODE_VOLATILE,
)


class Volatile:
    """Filter to volatile modes only.
    
    This transform keeps only items where state == MODE_VOLATILE
    or is_volatile == True.
    
    Example:
        observer = Volatile()(Modes())
        result = observer.observe(state)
        
        # Or chain directly:
        result = Volatile().observe(mode_result)
    """
    
    def __call__(self, inner: ObserverProtocol) -> _VolatileObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that filters to volatile items
        """
        return _VolatileObserver(inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Apply volatile filter directly to an ObservationResult.
        
        Args:
            state: ObservationResult to filter
            **kwargs: Ignored
        
        Returns:
            ObservationResult with only volatile items
        """
        if not isinstance(state, ObservationResult):
            return ObservationResult(data={"items": []}, source="volatile")
        
        items = [
            item for item in state.items
            if item.get("is_volatile", False) or 
               item.get("state") == MODE_VOLATILE
        ]
        
        return ObservationResult(
            data={**state.data, "items": items, "filter": "volatile"},
            source=state.source,
            _items=items,
        )


class _VolatileObserver:
    """Internal observer that wraps another observer with volatile filter."""
    
    def __init__(self, inner: ObserverProtocol):
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe with the inner observer, then filter to volatile."""
        result = self.inner.observe(state, **kwargs)
        
        items = [
            item for item in result.items
            if item.get("is_volatile", False) or 
               item.get("state") == MODE_VOLATILE
        ]
        
        return ObservationResult(
            data={**result.data, "items": items, "filter": "volatile"},
            source=result.source,
            _items=items,
        )
