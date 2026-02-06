"""Modes primitive observer.

Extracts ω-mode state from the simulation, providing access to
frequencies, amplitudes, gate widths, conflict scores, and mode state.
"""

from __future__ import annotations

from sensorium.observers.types import (
    ObservationResult,
    MODE_VOLATILE,
    MODE_STABLE,
    MODE_CRYSTALLIZED,
    to_numpy,
)


class Modes:
    """Extract ω-mode state from the simulation.

    Returns an ObservationResult with items representing each active mode.
    Each item contains: index, frequency, amplitude, gate_width, conflict, state, phase.

    Example:
        result = Modes().observe(state)
        crystallized = result.where(lambda m: m["state"] == MODE_CRYSTALLIZED)
        top_5 = result.top_k(5, by="amplitude")
    """

    def __init__(self, min_amplitude: float = 1e-6):
        """Initialize the modes observer.

        Args:
            min_amplitude: Minimum amplitude to consider a mode "active"
        """

        self.min_amplitude = float(min_amplitude)

    def observe(
        self,
        state: ObservationResult | dict | None = None,
        **kwargs,
    ) -> ObservationResult:
        """Extract mode data from state."""
        if isinstance(state, ObservationResult):
            raw_state = state.data
        elif isinstance(state, dict):
            raw_state = state
        else:
            return ObservationResult(data={"items": []}, source="modes")

        frequencies = raw_state.get("frequencies")
        amplitudes = raw_state.get("amplitudes")
        gate_widths = raw_state.get("gate_widths")
        conflict = raw_state.get("conflict")
        mode_state = raw_state.get("mode_state")
        phases = raw_state.get("phases")

        if amplitudes is None:
            return ObservationResult(data={"items": []}, source="modes")

        amplitudes_np = to_numpy(amplitudes)
        active_mask = amplitudes_np > self.min_amplitude
        active_indices = active_mask.nonzero()[0]

        items: list[dict] = []
        for idx in active_indices:
            item: dict = {
                "index": int(idx),
                "amplitude": float(amplitudes_np[idx]),
            }

            if frequencies is not None:
                item["frequency"] = float(to_numpy(frequencies)[idx])
            if gate_widths is not None:
                item["gate_width"] = float(to_numpy(gate_widths)[idx])
            if conflict is not None:
                item["conflict"] = float(to_numpy(conflict)[idx])
            if mode_state is not None:
                state_val = int(to_numpy(mode_state)[idx])
                item["state"] = state_val
                item["is_volatile"] = state_val == MODE_VOLATILE
                item["is_stable"] = state_val == MODE_STABLE
                item["is_crystallized"] = state_val == MODE_CRYSTALLIZED
            if phases is not None:
                item["phase"] = float(to_numpy(phases)[idx])

            items.append(item)

        data = {
            "items": items,
            "n_active": len(items),
            "n_total": len(amplitudes_np),
        }

        if mode_state is not None:
            state_np = to_numpy(mode_state)
            active_states = state_np[active_mask]
            data["n_volatile"] = int((active_states == MODE_VOLATILE).sum())
            data["n_stable"] = int((active_states == MODE_STABLE).sum())
            data["n_crystallized"] = int((active_states == MODE_CRYSTALLIZED).sum())

        return ObservationResult(data=data, source="modes", _items=items)

    # -------------------------------------------------------------------------
    # Fluent convenience methods
    # -------------------------------------------------------------------------

    def crystallized(self) -> _FilteredModes:
        """Return observer that filters to crystallized modes only."""

        return _FilteredModes(self, lambda m: m.get("is_crystallized", False))

    def stable(self) -> _FilteredModes:
        """Return observer that filters to stable modes only."""

        return _FilteredModes(self, lambda m: m.get("is_stable", False))

    def volatile(self) -> _FilteredModes:
        """Return observer that filters to volatile modes only."""

        return _FilteredModes(self, lambda m: m.get("is_volatile", False))


class _FilteredModes:
    """Helper class for filtered mode observations."""

    def __init__(self, parent: Modes, predicate):
        self.parent = parent
        self.predicate = predicate

    def observe(self, state: ObservationResult | dict | None = None, **kwargs) -> ObservationResult:
        result = self.parent.observe(state, **kwargs)
        return result.where(self.predicate)

    def count(self) -> _CountingModes:
        """Return observer that counts matching modes."""

        return _CountingModes(self)


class _CountingModes:
    """Helper class that counts modes."""

    def __init__(self, parent: _FilteredModes):
        self.parent = parent

    def observe(self, state: ObservationResult | dict | None = None, **kwargs) -> int:
        result = self.parent.observe(state, **kwargs)
        return result.count()

