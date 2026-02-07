"""Instrument protocol for the Sensorium Manifold.

Defines a common interface for instrumentation of the Manifold.
"""

from typing import Any, Protocol


class InstrumentProtocol(Protocol):
    def update(self, state: dict[str, Any]) -> None:
        """Update the instrument with the current (post-step) simulation state."""
        raise NotImplementedError("Subclasses must implement this method")
