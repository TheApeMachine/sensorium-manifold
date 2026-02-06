from __future__ import annotations

from collections import deque
import numpy as np


class EnergyPlot:
    """Energy conservation chart for the simulation dashboard."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self._step = 0
        self._history = {
            "step": deque(maxlen=500),
            "eint": deque(maxlen=500),
            "eheat": deque(maxlen=500),
            "ekin": deque(maxlen=500),
        }
        # Pre-create artists; update via set_data() (no ax.clear() per frame).
        (self._line_eint,) = ax.plot([], [], color="#2980b9", lw=1.5, label="E_mode")
        (self._line_eheat,) = ax.plot([], [], color="#c0392b", lw=1.5, label="E_heat")
        (self._line_ekin,) = ax.plot([], [], color="#7f8c8d", lw=1.5, label="E_kin")
        (self._line_total,) = ax.plot([], [], color="#8e44ad", ls="--", lw=1.5, label="Total")
        self._temp_text = ax.text(
            0.99,
            0.02,
            "",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            family="monospace",
        )
        ax.set_title("Energy Conservation", fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Step", fontsize=8)
        self._legend = ax.legend(fontsize=7, loc="upper left")

    def update(self, state: dict) -> None:
        """Update the energy chart from state dict."""
        self._step += 1
        
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v
        
        energy = get("energies", np.array([]))  # oscillator / mode energy (not thermal)
        heat = get("heats", np.array([]))
        masses = get("masses")
        velocities = get("velocities", np.zeros((0, 3)))
        c_v = state.get("c_v", None)
        if c_v is not None and hasattr(c_v, "item"):
            c_v = float(c_v.item())
        if c_v is not None:
            c_v = float(c_v)
        
        e_mode = float(energy.sum()) if len(energy) > 0 else 0.0
        eheat = float(heat.sum()) if len(heat) > 0 else 0.0
        
        m_eff = masses if masses is not None else energy
        v2 = (velocities ** 2).sum(axis=1) if len(velocities) > 0 else np.array([])
        ekin = float(0.5 * np.sum(m_eff * v2)) if len(v2) > 0 else 0.0
        
        self._history["step"].append(self._step)
        self._history["eint"].append(e_mode)
        self._history["eheat"].append(eheat)
        self._history["ekin"].append(ekin)
        steps = list(self._history["step"])
        if len(steps) > 1:
            eint_hist = np.asarray(self._history["eint"], dtype=np.float64)
            eheat_hist = np.asarray(self._history["eheat"], dtype=np.float64)
            ekin_hist = np.asarray(self._history["ekin"], dtype=np.float64)
            total_hist = eint_hist + eheat_hist + ekin_hist

            self._line_eint.set_data(steps, eint_hist)
            self._line_eheat.set_data(steps, eheat_hist)
            self._line_ekin.set_data(steps, ekin_hist)
            self._line_total.set_data(steps, total_hist)
        else:
            # Not enough history yet: clear lines
            self._line_eint.set_data([], [])
            self._line_eheat.set_data([], [])
            self._line_ekin.set_data([], [])
            self._line_total.set_data([], [])

        # Add a small temperature readout (mean over particles) when c_v is available.
        temp_str = ""
        if c_v is not None and c_v > 0.0 and masses is not None and len(heat) > 0:
            m = masses
            if hasattr(m, "astype"):
                m = m.astype(np.float64)
            h = heat.astype(np.float64)
            denom = m * float(c_v)
            valid = denom > 0
            if np.any(valid):
                T_mean = float(np.mean(h[valid] / denom[valid]))
                temp_str = f"T̄ ≈ {T_mean:.3g}"
        self._temp_text.set_text(temp_str)

        # Autoscale to new data without clearing the axes.
        self.ax.relim()
        self.ax.autoscale_view()
