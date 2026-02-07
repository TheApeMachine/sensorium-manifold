"""Simple state-history instrument for Manifold runs."""

from __future__ import annotations

from typing import Any


class StateHistoryInstrument:
    """Capture per-step state snapshots for observer-side analysis.

    This instrument is intentionally minimal:
    - append snapshots to an in-memory list (`history`)
    - optional downsampling via `sample_every`
    - optional cap via `max_frames`
    """

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def update(self, state: dict[str, Any]) -> None:
        self.history.append(state)