from __future__ import annotations

from collections import deque

import numpy as np


class SettlingPlot:
    """Right-column settling diagnostics.

    Goal: make "is it settling?" visible at a glance, without relying on logs.

    Signals:
    - psi_delta_rms: RMS change in Ψ(ω) per frame (→ 0 when field converges)
    - cfl_step: dt * max_characteristic_rate (should be O(CFL) and stabilize)
    - kuramoto_R: global phase order in Ψ (0..1)
    - heat_sum (optional): total particle heat (can be noisy; included for context)
    """

    def __init__(self, ax) -> None:
        self.ax = ax
        self._hist = {
            "step": deque(maxlen=500),
            "psi_delta": deque(maxlen=500),
            "cfl_step": deque(maxlen=500),
            "R": deque(maxlen=500),
            "heat": deque(maxlen=500),
        }
        self._prev_psi: np.ndarray | None = None

        ax.set_title("Settling dynamics", fontsize=10, pad=6)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yscale("log")

        # Left axis (log): convergence signals.
        (self._ln_dpsi,) = ax.plot([], [], color="#f39c12", lw=1.5, label="||ΔΨ|| (rms)")
        (self._ln_cfl,) = ax.plot([], [], color="#3498db", lw=1.5, label="dt·rate (CFL)")

        # Right axis: Kuramoto order parameter.
        self._ax2 = ax.twinx()
        self._ax2.tick_params(labelsize=7)
        (self._ln_R,) = self._ax2.plot([], [], color="#2ecc71", lw=1.5, label="R (Kuramoto)")
        self._ax2.set_ylim(0.0, 1.05)
        self._ax2.set_ylabel("R", fontsize=8)

        # Small status text.
        self._status = ax.text(
            0.99,
            0.02,
            "",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            family="monospace",
        )

        # Combined legend.
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color="#f39c12", lw=2, label="||ΔΨ||"),
            Line2D([0], [0], color="#3498db", lw=2, label="dt·rate"),
            Line2D([0], [0], color="#2ecc71", lw=2, label="R"),
        ]
        ax.legend(
            handles=handles,
            fontsize=7,
            loc="upper left",
            framealpha=0.35,
        )
        self._frame: dict | None = None

    def ingest(self, state: dict) -> None:
        def to_numpy(v, default=None):
            if v is None:
                return default
            if hasattr(v, "detach"):
                return v.detach().cpu().numpy()
            return v

        def scalar(v, default=0.0) -> float:
            if v is None:
                return float(default)
            if hasattr(v, "item"):
                return float(v.item())
            try:
                return float(v)
            except Exception:
                return float(default)

        step = scalar(state.get("step", 0), 0.0)
        psi_r = to_numpy(state.get("psi_real"), None)
        psi_i = to_numpy(state.get("psi_imag"), None)

        # CFL-per-step: dt * max_rate (dimensionless).
        dt = scalar(state.get("dt", 0.0), 0.0)
        max_rate = scalar(state.get("cfl_max_rate", 0.0), 0.0)
        cfl_step = float(dt * max_rate) if (dt > 0.0 and max_rate > 0.0) else 0.0

        # Kuramoto R + field delta.
        R = 0.0
        dpsi = 0.0
        if psi_r is not None and psi_i is not None:
            psi = psi_r.astype(np.float64) + 1j * psi_i.astype(np.float64)
            if psi.size > 0:
                phases = np.angle(psi)
                R = float(np.abs(np.mean(np.exp(1j * phases))))
            if self._prev_psi is not None and self._prev_psi.shape == psi.shape and psi.size > 0:
                d = psi - self._prev_psi
                dpsi = float(np.sqrt(np.mean(d.real * d.real + d.imag * d.imag)))
            self._prev_psi = psi

        heat = to_numpy(state.get("heats"), None)
        heat_sum = float(np.sum(heat)) if isinstance(heat, np.ndarray) and heat.size else 0.0

        h = self._hist
        h["step"].append(step)
        h["psi_delta"].append(max(dpsi, 1e-12))  # keep log scale happy
        h["cfl_step"].append(max(cfl_step, 1e-12))
        h["R"].append(R)
        h["heat"].append(heat_sum)

        steps = np.asarray(h["step"], dtype=np.float64)
        if steps.size < 2:
            self._frame = {
                "ready": False,
                "status": "",
            }
            return

        dpsi_a = np.asarray(h["psi_delta"], dtype=np.float64)
        cfl_a = np.asarray(h["cfl_step"], dtype=np.float64)
        R_a = np.asarray(h["R"], dtype=np.float64)

        # Status line (simple heuristic).
        tail = min(25, len(dpsi_a))
        dpsi_recent = float(np.median(dpsi_a[-tail:])) if tail > 0 else float("nan")
        self._frame = {
            "ready": True,
            "steps": steps,
            "dpsi": dpsi_a,
            "cfl": cfl_a,
            "R": R_a,
            "status": f"median(||ΔΨ||)[{tail}]={dpsi_recent:.2e}  R={R_a[-1]:.2f}",
        }

    def render(self) -> None:
        frame = self._frame
        if frame is None:
            return
        if not bool(frame.get("ready", False)):
            self._status.set_text(frame.get("status", ""))
            return

        self._ln_dpsi.set_data(frame["steps"], frame["dpsi"])
        self._ln_cfl.set_data(frame["steps"], frame["cfl"])
        self._ln_R.set_data(frame["steps"], frame["R"])
        self._status.set_text(frame["status"])

        self.ax.relim()
        self.ax.autoscale_view()
        self._ax2.relim()
        self._ax2.autoscale_view(scalex=False, scaley=False)

    def update(self, state: dict) -> None:
        self.ingest(state)
        self.render()
