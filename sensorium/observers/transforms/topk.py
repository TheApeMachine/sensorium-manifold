"""TopK transform observer.

Selects the top-k items by a specified field.
"""

from __future__ import annotations

from sensorium.observers.types import ObservationResult, ObserverProtocol


class TopK:
    """Select top-k items by a field value.
    
    This transform sorts items by the specified field (descending)
    and takes the top k.
    
    Example:
        observer = TopK(5, by="amplitude")(Modes())
        result = observer.observe(state)
        
        # Or use directly:
        result = TopK(10, by="energy").observe(particle_result)
    """
    
    def __init__(self, k: int, by: str, descending: bool = True):
        """Initialize the top-k transform.
        
        Args:
            k: Number of items to keep
            by: Field name to sort by
            descending: If True (default), highest values first
        """
        self.k = k
        self.by = by
        self.descending = descending
    
    def __call__(self, inner: ObserverProtocol) -> _TopKObserver:
        """Wrap an inner observer.
        
        Args:
            inner: The observer to wrap
        
        Returns:
            A new observer that applies top-k to the inner observer's result
        """
        return _TopKObserver(self.k, self.by, self.descending, inner)
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Apply top-k directly to an ObservationResult.
        
        Args:
            state: ObservationResult to apply top-k to
            **kwargs: Ignored
        
        Returns:
            ObservationResult with top-k items
        """
        if not isinstance(state, ObservationResult):
            return ObservationResult(data={"items": []}, source="topk")
        
        items = sorted(
            state.items,
            key=lambda x: x.get(self.by, 0),
            reverse=self.descending,
        )[:self.k]
        
        return ObservationResult(
            data={**state.data, "items": items, "k": self.k, "sort_by": self.by},
            source=state.source,
            _items=items,
        )


class _TopKObserver:
    """Internal observer that wraps another observer with top-k selection."""
    
    def __init__(
        self,
        k: int,
        by: str,
        descending: bool,
        inner: ObserverProtocol,
    ):
        self.k = k
        self.by = by
        self.descending = descending
        self.inner = inner
    
    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Observe with the inner observer, then apply top-k."""
        result = self.inner.observe(state, **kwargs)
        
        items = sorted(
            result.items,
            key=lambda x: x.get(self.by, 0),
            reverse=self.descending,
        )[:self.k]
        
        return ObservationResult(
            data={**result.data, "items": items, "k": self.k, "sort_by": self.by},
            source=result.source,
            _items=items,
        )
