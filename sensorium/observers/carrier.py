from sensorium.observers.base import ObserverProtocol
from sensorium.manifold.carriers import CarrierState
from typing import Optional


class CarrierObserver(ObserverProtocol):
    def __init__(self, carrier_state: Optional[CarrierState] = None):
        self._carrier_state = carrier_state
        self._current_observation = None

    @property
    def most_coherent_carrier(self):
        conflict = self.conflict
        if conflict is None or len(conflict) == 0:
            return None
        min_conflict = float('inf')
        min_conflict_index = None
        for i in range(len(conflict)):
            if conflict[i] < min_conflict:
                min_conflict = conflict[i]
                min_conflict_index = i
        return min_conflict_index

    def observe(self, observation=None, **kwargs):
        # If observation is provided (e.g., from spectral.step()), use it
        if observation is not None and isinstance(observation, dict):
            self._current_observation = observation
            # Extract data from observation dict
            frequencies = observation.get("frequencies")
            conflict = observation.get("conflict")
            
            # Determine done_thinking based on observation
            done_thinking = False
            if conflict is not None and len(conflict) > 0:
                # Done thinking if we have at least one carrier with low conflict
                min_conflict = float('inf')
                for i in range(len(conflict)):
                    if conflict[i] < min_conflict:
                        min_conflict = conflict[i]
                # Consider done if minimum conflict is below a threshold
                done_thinking = min_conflict < 0.1 if min_conflict != float('inf') else False
            
            return {
                "frequencies": frequencies,
                "gate_widths": observation.get("gate_widths"),
                "amplitudes": observation.get("amplitudes"),
                "phases": observation.get("phases"),
                "conflict": conflict,
                "osc_phase": observation.get("osc_phase"),
                "osc_energy": observation.get("osc_energy"),
                "carrier_state": observation.get("carrier_state"),
                "carrier_age": observation.get("carrier_age"),
                "done_thinking": done_thinking,
                "idle_time": observation.get("idle_time", 0),
            }
        
        # Fallback to _carrier_state if available
        if self._carrier_state is not None:
            return {
                "frequencies": self._carrier_state.frequencies,
                "gate_widths": self._carrier_state.gate_widths,
                "amplitudes": self._carrier_state.amplitudes,
                "phases": self._carrier_state.phases,
                "conflict": self._carrier_state.conflict,
                "osc_phase": self._carrier_state.osc_phase,
                "osc_energy": self._carrier_state.osc_energy,
                "carrier_state": self._carrier_state.carrier_state,
                "carrier_age": self._carrier_state.carrier_age,
                "done_thinking": self.done_thinking,
                "idle_time": getattr(self._carrier_state, 'idle_time', 0),
            }
        
        # Default empty observation
        return {"done_thinking": True}

    @property
    def frequencies(self):
        if self._current_observation:
            return self._current_observation.get("frequencies")
        return self._carrier_state.frequencies if self._carrier_state else None

    @property
    def gate_widths(self):
        if self._current_observation:
            return self._current_observation.get("gate_widths")
        return self._carrier_state.gate_widths if self._carrier_state else None

    @property
    def amplitudes(self):
        if self._current_observation:
            return self._current_observation.get("amplitudes")
        return self._carrier_state.amplitudes if self._carrier_state else None

    @property
    def phases(self):
        if self._current_observation:
            return self._current_observation.get("phases")
        return self._carrier_state.phases if self._carrier_state else None

    @property
    def conflict(self):
        if self._current_observation:
            return self._current_observation.get("conflict")
        return self._carrier_state.conflict if self._carrier_state else None

    @property
    def osc_phase(self):
        if self._current_observation:
            return self._current_observation.get("osc_phase")
        return self._carrier_state.osc_phase if self._carrier_state else None

    @property
    def osc_energy(self):
        if self._current_observation:
            return self._current_observation.get("osc_energy")
        return self._carrier_state.osc_energy if self._carrier_state else None

    @property
    def carrier_state(self):
        if self._current_observation:
            return self._current_observation.get("carrier_state")
        return self._carrier_state.carrier_state if self._carrier_state else None

    @property
    def carrier_age(self):
        if self._current_observation:
            return self._current_observation.get("carrier_age")
        return self._carrier_state.carrier_age if self._carrier_state else None

    @property
    def done_thinking(self):
        return self.most_coherent_carrier is not None

    @property
    def idle_time(self):
        if self._current_observation:
            return self._current_observation.get("idle_time", 0)
        return getattr(self._carrier_state, 'idle_time', 0) if self._carrier_state else 0