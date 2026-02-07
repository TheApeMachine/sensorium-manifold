"""Spectrogram + amplitude margin visualization."""

from __future__ import annotations

from collections import deque

import numpy as np

_MAX_MARKERS = 64
_MAX_MARGIN_BARS = 64


class SpectrogramPlot:
    """Rolling spectrogram with amplitude margin bars."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self._max_history = 100
        self._history: deque[np.ndarray] = deque(maxlen=self._max_history)
        self._omega_range: tuple[float, float] = (0.0, 1.0)

        self._im = ax.imshow(
            np.zeros((1, 1), dtype=np.float64),
            aspect="auto", origin="lower",
            extent=[0, 1, 0.0, 1.0],
            cmap="magma", vmin=-8.0, vmax=0.0,
            interpolation="bilinear",
        )
        ax.set_xlabel("Time (steps)", fontsize=7)
        ax.set_ylabel("omega", fontsize=7)
        ax.tick_params(labelsize=5)

        # Pre-allocate marker lines
        self._markers: list[object] = []
        for _ in range(_MAX_MARKERS):
            ln = ax.axhline(0, xmin=0.92, xmax=1.0, color='white', lw=1, alpha=0.0)
            self._markers.append(ln)

        # Amplitude margin inset
        self._ax_margin = ax.inset_axes([0.89, 0.0, 0.10, 1.0])
        self._ax_margin.set_xticks([])
        self._ax_margin.set_yticks([])
        for spine in self._ax_margin.spines.values():
            spine.set_visible(False)
        self._margin_bars = self._ax_margin.barh(
            np.arange(_MAX_MARGIN_BARS, dtype=np.float64),
            np.zeros(_MAX_MARGIN_BARS, dtype=np.float64),
            height=0.8, color='#7f8c8d', alpha=0.0,
        )
        self._frame: dict | None = None

    def ingest(self, state: dict) -> None:
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v

        psi_amp = get("psi_amplitude")
        omega = get("omega_lattice")
        mode_state = get("mode_state")

        if psi_amp is None or omega is None:
            self._frame = {
                "title": "spectrogram: awaiting data",
                "spec_db": np.zeros((1, 1), dtype=np.float64),
                "extent": [0.0, 1.0, self._omega_range[0], self._omega_range[1]],
                "clim": (-8.0, 0.0),
                "marker_specs": [],
                "bar_specs": [],
                "margin_xlim": (0.0, 1.1),
                "margin_ylim": self._omega_range,
            }
            return

        amp = psi_amp.astype(np.float64)
        omega_arr = omega.astype(np.float64)
        M = len(amp)

        if len(omega_arr) > 1:
            self._omega_range = (float(omega_arr.min()), float(omega_arr.max()))

        self._history.append(amp.copy())

        if len(self._history) < 2:
            self._frame = {
                "title": "spectrogram: accumulating...",
                "spec_db": np.zeros((1, 1), dtype=np.float64),
                "extent": [0.0, 1.0, self._omega_range[0], self._omega_range[1]],
                "clim": (-8.0, 0.0),
                "marker_specs": [],
                "bar_specs": [],
                "margin_xlim": (0.0, 1.1),
                "margin_ylim": self._omega_range,
            }
            return

        spec = np.array(list(self._history))
        T, _ = spec.shape
        spec_db = np.log10(spec + 1e-8)
        mask_valid = spec_db > np.log10(1e-8)
        vmin = float(np.percentile(spec_db[mask_valid], 5)) if np.any(mask_valid) else -8
        vmax = float(np.percentile(spec_db, 98))
        if not (vmax > vmin):
            vmax = vmin + 1.0

        # Markers
        marker_specs: list[tuple[float, str, float, float, str]] = []
        ms = mode_state.astype(np.int32) if mode_state is not None else np.zeros(M, dtype=np.int32)

        for om in omega_arr[ms == 2]:
            if len(marker_specs) < _MAX_MARKERS:
                marker_specs.append((float(om), '#f39c12', 2.0, 0.9, '-'))

        for om in omega_arr[ms == 1]:
            if len(marker_specs) < _MAX_MARKERS:
                marker_specs.append((float(om), '#3498db', 1.5, 0.7, '-'))

        current_amp = spec[-1]
        top_idx = np.argsort(current_amp)[-3:][::-1]
        p70 = float(np.percentile(current_amp, 70))
        for idx in top_idx:
            if current_amp[idx] > p70 and len(marker_specs) < _MAX_MARKERS:
                marker_specs.append((float(omega_arr[idx]), '#ccc', 0.5, 0.3, '--'))

        # Margin bars
        state_colors = {0: '#7f8c8d', 1: '#3498db', 2: '#f39c12'}
        amp_max = max(float(np.max(amp)), 1e-8)
        bar_specs: list[tuple[float, float, float, str, float]] = []
        for i in range(min(M, _MAX_MARGIN_BARS)):
            y = float(omega_arr[i]) if i < len(omega_arr) else float(i)
            width = float(amp[i]) / amp_max
            height = (self._omega_range[1] - self._omega_range[0]) / max(M, 1) * 0.85
            color = state_colors.get(int(ms[i]), '#7f8c8d')
            bar_specs.append((y, width, height, color, 0.6))

        total_power = float(np.sum(amp ** 2))
        n_crystal = int(np.sum(ms == 2))
        active_modes = int(np.sum(amp > np.median(amp)))
        self._frame = {
            "title": f'spectrogram | pwr={total_power:.1f} | {active_modes}/{M} active | {n_crystal} crystal',
            "spec_db": spec_db.T,
            "extent": [0, T, self._omega_range[0], self._omega_range[1]],
            "clim": (vmin, vmax),
            "marker_specs": marker_specs,
            "bar_specs": bar_specs,
            "margin_xlim": (0.0, 1.1),
            "margin_ylim": self._omega_range,
        }

    def render(self) -> None:
        frame = self._frame
        if frame is None:
            return

        self._im.set_data(frame["spec_db"])
        self._im.set_extent(frame["extent"])
        self._im.set_clim(vmin=frame["clim"][0], vmax=frame["clim"][1])

        marker_specs = frame["marker_specs"]
        for i, ln in enumerate(self._markers):
            if i < len(marker_specs):
                y, color, lw, alpha, linestyle = marker_specs[i]
                ln.set_ydata([y, y])
                ln.set_color(color)
                ln.set_linewidth(lw)
                ln.set_alpha(alpha)
                ln.set_linestyle(linestyle)
            else:
                ln.set_alpha(0.0)

        bar_specs = frame["bar_specs"]
        for i, rect in enumerate(self._margin_bars):
            if i < len(bar_specs):
                y, width, height, color, alpha = bar_specs[i]
                rect.set_y(y)
                rect.set_width(width)
                rect.set_height(height)
                rect.set_color(color)
                rect.set_alpha(alpha)
            else:
                rect.set_alpha(0.0)

        self._ax_margin.set_xlim(*frame["margin_xlim"])
        self._ax_margin.set_ylim(*frame["margin_ylim"])
        self.ax.set_title(frame["title"], fontsize=7, pad=4)

    def update(self, state: dict) -> None:
        self.ingest(state)
        self.render()
