"""Crystallized transform observer.

Filters to only crystallized modes.
"""

from __future__ import annotations

from sensorium.observers.types import (
    ObservationResult,
    ObserverProtocol,
    MODE_CRYSTALLIZED,
)


class Crystallized:
    """Filter to crystallized modes only.
    
    This transform keeps only items where state == MODE_CRYSTALLIZED
    or is_crystallized == True.
    
    Example:
        observer = Crystallized()(Modes())
        result = observer.observe(state)
        
        # Or chain directly:
        result = Crystallized().observe(mode_result)
    """
    
    def __call__(self, inner: ObserverProtocol) -> _CrystallizedObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that filters to crystallized items
        """
        return _CrystallizedObserver(inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Apply crystallized filter directly to an ObservationResult.
        
        Args:
            state: ObservationResult to filter
            **kwargs: Ignored
        
        Returns:
            ObservationResult with only crystallized items
        """
        if not isinstance(state, ObservationResult):
            return ObservationResult(data={"items": []}, source="crystallized")
        
        items = [
            item for item in state.items
            if item.get("is_crystallized", False) or 
               item.get("state") == MODE_CRYSTALLIZED
        ]
        
        return ObservationResult(
            data={**state.data, "items": items, "filter": "crystallized"},
            source=state.source,
            _items=items,
        )


class _CrystallizedObserver:
    """Internal observer that wraps another observer with crystallized filter."""
    
    def __init__(self, inner: ObserverProtocol):
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe with the inner observer, then filter to crystallized."""
        result = self.inner.observe(state, **kwargs)
        
        items = [
            item for item in result.items
            if item.get("is_crystallized", False) or 
               item.get("state") == MODE_CRYSTALLIZED
        ]
        
        return ObservationResult(
            data={**result.data, "items": items, "filter": "crystallized"},
            source=result.source,
            _items=items,
        )
