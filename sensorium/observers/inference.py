"""Inference observer for the manifold.

The InferenceObserver is the central data store for experiment observations.
It accumulates results across multiple observe() calls and provides a
query interface for projectors to extract data by field name.

Design:
- Observers compute metrics and return dicts
- InferenceObserver accumulates these into a results list
- Projectors query the InferenceObserver for specific fields
- No manual data extraction needed in experiments

Example:
    # Setup
    inference = InferenceObserver([
        SpatialClustering(),
        CompressionRatio(),
    ])

    # Run multiple times (e.g., different collision rates)
    for dataset in datasets:
        manifold.add_dataset(dataset.generate)
        state = manifold.run()
        inference.observe(state, collision_rate=rate)  # Add metadata

    # Projector queries directly
    projector.project(inference)  # Gets data via inference.get("field_name")
"""

from __future__ import annotations

from typing import Callable, Sequence

from sensorium.manifold import Manifold
from sensorium.observers.base import ObserverProtocol


class InferenceObserver(ObserverProtocol):
    """The InferenceObserver is an observer pipeline

    Runs a sequence of observers on the state and accumulates the results.
    The easiest way to use this is to pass it the SQLObserver, which allows
    you to write SQL queries in a high-level mode, that abstract away the
    physics and data structures.
    """

    def __init__(
        self,
        manifold: Manifold,
        observers: Sequence[ObserverProtocol | Callable] | None = None,
    ):
        self.observers = observers or []

    def observe(self, state: dict, **kwargs) -> dict:
        """Observe the state and return a result."""
        for observer in self.observers:
            result = observer.observe(state, **kwargs)
            self.results.append(result)

        return self.results