from __future__ import annotations

from collections import deque
import numpy as np


class SigmaPlot:
    """Mode linewidth (γ_k) statistics chart for the simulation dashboard."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self._step = 0
        self._history = {
            "step": deque(maxlen=500),
            "mean": deque(maxlen=500),
            "p10": deque(maxlen=500),
            "p90": deque(maxlen=500),
            "min": deque(maxlen=500),
            "max": deque(maxlen=500),
        }
        # Pre-create artists; update via set_data / polygon vertices.
        (self._line_mean,) = ax.plot([], [], color="#16a085", lw=1.6, label="γ mean")
        (self._line_min,) = ax.plot([], [], color="#16a085", lw=0.8, alpha=0.5, ls=":")
        (self._line_max,) = ax.plot([], [], color="#16a085", lw=0.8, alpha=0.5, ls=":")
        self._poly_band = ax.fill_between([], [], [], color="#16a085", alpha=0.20, label="γ p10–p90")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Mode linewidth (γ_k)", fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Step", fontsize=8)

    def update(self, state: dict) -> None:
        """Update the sigma chart from state dict."""
        self._step += 1
        
        gamma = state.get("mode_linewidth")
        if gamma is not None and hasattr(gamma, 'detach'):
            gamma = gamma.detach().cpu().numpy()
        
        if gamma is not None and len(gamma) > 0:
            self._history["step"].append(self._step)
            self._history["mean"].append(float(np.mean(gamma)))
            self._history["p10"].append(float(np.percentile(gamma, 10)))
            self._history["p90"].append(float(np.percentile(gamma, 90)))
            self._history["min"].append(float(np.min(gamma)))
            self._history["max"].append(float(np.max(gamma)))

        steps = list(self._history["step"])
        if len(steps) > 1:
            s_mean = np.array(list(self._history["mean"]))
            s_p10 = np.array(list(self._history["p10"]))
            s_p90 = np.array(list(self._history["p90"]))
            s_min = np.array(list(self._history["min"]))
            s_max = np.array(list(self._history["max"]))
            
            valid = np.isfinite(s_mean)
            if valid.any():
                st = np.array(steps)
                x = st[valid].astype(np.float64)
                y_mean = s_mean[valid].astype(np.float64)
                y_p10 = s_p10[valid].astype(np.float64)
                y_p90 = s_p90[valid].astype(np.float64)
                y_min = s_min[valid].astype(np.float64)
                y_max = s_max[valid].astype(np.float64)

                self._line_mean.set_data(x, y_mean)
                self._line_min.set_data(x, y_min)
                self._line_max.set_data(x, y_max)

                # Update p10–p90 band polygon
                if x.size:
                    verts = np.concatenate(
                        [
                            np.column_stack([x, y_p10]),
                            np.column_stack([x[::-1], y_p90[::-1]]),
                            np.array([[x[0], y_p10[0]]], dtype=np.float64),
                        ],
                        axis=0,
                    )
                    try:
                        self._poly_band.get_paths()[0].vertices = verts
                    except Exception:
                        pass
        else:
            self._line_mean.set_data([], [])
            self._line_min.set_data([], [])
            self._line_max.set_data([], [])
