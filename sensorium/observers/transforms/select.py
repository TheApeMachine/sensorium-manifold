"""Select transform observer.

Picks specific fields from observation items, reducing each item
to only the selected fields.
"""

from __future__ import annotations

from typing import Any

from sensorium.observers.types import ObservationResult, ObserverProtocol


class Select:
    """Select specific fields from observation items.
    
    This transform reduces each item to only contain the specified fields.
    Useful for extracting just the data you need.
    
    Example:
        observer = Select("index", "amplitude")(Modes())
        result = observer.observe(state)
        # Each item now only has "index" and "amplitude"
    """
    
    def __init__(self, *fields: str):
        """Initialize the select transform.
        
        Args:
            *fields: Field names to select from each item
        """
        if not fields:
            raise ValueError("Select requires at least one field")
        self.fields = fields
    
    def __call__(self, inner: ObserverProtocol) -> _SelectObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that selects fields from the inner observer's result
        """
        return _SelectObserver(self.fields, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Apply selection directly to an ObservationResult.
        
        Args:
            state: ObservationResult to select from
            **kwargs: Ignored
        
        Returns:
            ObservationResult with selected fields only
        """
        if not isinstance(state, ObservationResult):
            return ObservationResult(data={"items": []}, source="select")
        
        items = [
            {k: item.get(k) for k in self.fields if k in item}
            for item in state.items
        ]
        
        return ObservationResult(
            data={**state.data, "items": items},
            source=state.source,
            _items=items,
        )


class _SelectObserver:
    """Internal observer that wraps another observer with selection."""
    
    def __init__(self, fields: tuple[str, ...], inner: ObserverProtocol):
        self.fields = fields
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe with the inner observer, then select fields."""
        result = self.inner.observe(state, **kwargs)
        
        items = [
            {k: item.get(k) for k in self.fields if k in item}
            for item in result.items
        ]
        
        return ObservationResult(
            data={**result.data, "items": items},
            source=result.source,
            _items=items,
        )
