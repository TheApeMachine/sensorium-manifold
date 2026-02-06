from __future__ import annotations

from typing import Any, Optional

import torch

from sensorium.observers.base import ObserverProtocol


class ModeObserver(ObserverProtocol):
    """Normalize ω-mode (coherence field) outputs into a stable observer shape.

    Expected input keys (preferred):
    - frequencies, gate_widths, amplitudes, phases, conflict, mode_state

    Fallback keys (produced by `OmegaWaveDomain`):
    - omega_lattice, mode_linewidth, psi_amplitude, psi_phase, mode_conflict, mode_state
    - phase (oscillator phase), energy_osc (oscillator energy)
    """

    def __init__(self, *, done_conflict_threshold: float = 0.1, active_amp_threshold: float = 1e-6) -> None:
        self.done_conflict_threshold = float(done_conflict_threshold)
        self.active_amp_threshold = float(active_amp_threshold)

    def observe(self, observation: Any = None, **kwargs) -> dict:
        if not isinstance(observation, dict):
            return {"done_thinking": True}

        frequencies = observation.get("frequencies", observation.get("omega_lattice"))
        gate_widths = observation.get("gate_widths", observation.get("mode_linewidth"))
        amplitudes = observation.get("amplitudes", observation.get("psi_amplitude"))
        phases = observation.get("phases", observation.get("psi_phase"))
        conflict = observation.get("conflict", observation.get("mode_conflict"))
        mode_state = observation.get("mode_state")

        osc_phase = observation.get("osc_phase", observation.get("phase"))
        osc_energy = observation.get("osc_energy", observation.get("energy_osc", observation.get("energies")))

        done_thinking = False
        try:
            if conflict is not None and hasattr(conflict, "numel") and conflict.numel() > 0:
                # Consider only active modes (by amplitude) when available.
                if amplitudes is not None and hasattr(amplitudes, "numel") and amplitudes.numel() == conflict.numel():
                    active = amplitudes > float(self.active_amp_threshold)
                    if bool(active.any().detach().item()):
                        done_thinking = bool((conflict[active] < float(self.done_conflict_threshold)).any().detach().item())
                    else:
                        done_thinking = False
                else:
                    done_thinking = bool((conflict < float(self.done_conflict_threshold)).any().detach().item())
        except Exception:
            done_thinking = False

        out = {
            "frequencies": frequencies,
            "gate_widths": gate_widths,
            "amplitudes": amplitudes,
            "phases": phases,
            "conflict": conflict,
            "mode_state": mode_state,
            "osc_phase": osc_phase,
            "osc_energy": osc_energy,
            "done_thinking": done_thinking,
        }

        # Avoid shipping None-heavy dicts to downstream consumers.
        return {k: v for k, v in out.items() if v is not None}

