"""Spectrogram + amplitude margin visualization (dark theme)."""

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
        ax.set_xlabel("Time (steps)", fontsize=7, color='#aaa')
        ax.set_ylabel("omega", fontsize=7, color='#aaa')
        ax.tick_params(labelsize=5, colors='#aaa')

        # Pre-allocate marker lines
        self._markers: list[object] = []
        for _ in range(_MAX_MARKERS):
            ln = ax.axhline(0, xmin=0.92, xmax=1.0, color='white', lw=1, alpha=0.0)
            self._markers.append(ln)

        # Amplitude margin inset
        self._ax_margin = ax.inset_axes([0.89, 0.0, 0.10, 1.0])
        self._ax_margin.set_facecolor('none')
        self._ax_margin.patch.set_alpha(0.0)
        self._ax_margin.set_xticks([])
        self._ax_margin.set_yticks([])
        for spine in self._ax_margin.spines.values():
            spine.set_visible(False)
        self._margin_bars = self._ax_margin.barh(
            np.arange(_MAX_MARGIN_BARS, dtype=np.float64),
            np.zeros(_MAX_MARGIN_BARS, dtype=np.float64),
            height=0.8, color='#7f8c8d', alpha=0.0,
        )

    def update(self, state: dict) -> None:
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v

        psi_amp = get("psi_amplitude")
        omega = get("omega_lattice")
        mode_state = get("mode_state")

        if psi_amp is None or omega is None:
            self.ax.set_title("spectrogram: awaiting data", fontsize=8, color='#ccc')
            self._im.set_data(np.zeros((1, 1), dtype=np.float64))
            return

        amp = psi_amp.astype(np.float64)
        omega_arr = omega.astype(np.float64)
        M = len(amp)

        if len(omega_arr) > 1:
            self._omega_range = (float(omega_arr.min()), float(omega_arr.max()))

        self._history.append(amp.copy())

        if len(self._history) < 2:
            self.ax.set_title("spectrogram: accumulating...", fontsize=8, color='#ccc')
            return

        spec = np.array(list(self._history))
        T, _ = spec.shape
        spec_db = np.log10(spec + 1e-8)
        mask_valid = spec_db > np.log10(1e-8)
        vmin = float(np.percentile(spec_db[mask_valid], 5)) if np.any(mask_valid) else -8
        vmax = float(np.percentile(spec_db, 98))
        if not (vmax > vmin):
            vmax = vmin + 1.0

        self._im.set_data(spec_db.T)
        self._im.set_extent([0, T, self._omega_range[0], self._omega_range[1]])
        self._im.set_clim(vmin=vmin, vmax=vmax)

        # Markers
        marker_idx = 0
        ms = mode_state.astype(np.int32) if mode_state is not None else np.zeros(M, dtype=np.int32)

        for om in omega_arr[ms == 2]:
            if marker_idx < _MAX_MARKERS:
                ln = self._markers[marker_idx]
                ln.set_ydata([om, om])
                ln.set_color('#f39c12')
                ln.set_linewidth(2)
                ln.set_alpha(0.9)
                ln.set_linestyle('-')
                marker_idx += 1

        for om in omega_arr[ms == 1]:
            if marker_idx < _MAX_MARKERS:
                ln = self._markers[marker_idx]
                ln.set_ydata([om, om])
                ln.set_color('#3498db')
                ln.set_linewidth(1.5)
                ln.set_alpha(0.7)
                ln.set_linestyle('-')
                marker_idx += 1

        current_amp = spec[-1]
        top_idx = np.argsort(current_amp)[-3:][::-1]
        p70 = float(np.percentile(current_amp, 70))
        for idx in top_idx:
            if current_amp[idx] > p70 and marker_idx < _MAX_MARKERS:
                ln = self._markers[marker_idx]
                ln.set_ydata([omega_arr[idx], omega_arr[idx]])
                ln.set_color('#ccc')
                ln.set_linewidth(0.5)
                ln.set_alpha(0.3)
                ln.set_linestyle('--')
                marker_idx += 1

        for i in range(marker_idx, _MAX_MARKERS):
            self._markers[i].set_alpha(0.0)

        # Margin bars
        state_colors = {0: '#7f8c8d', 1: '#3498db', 2: '#f39c12'}
        amp_max = max(float(np.max(amp)), 1e-8)
        for i, rect in enumerate(self._margin_bars):
            if i < M:
                rect.set_y(omega_arr[i] if i < len(omega_arr) else i)
                rect.set_width(float(amp[i]) / amp_max)
                rect.set_height((self._omega_range[1] - self._omega_range[0]) / max(M, 1) * 0.85)
                rect.set_color(state_colors.get(int(ms[i]), '#7f8c8d'))
                rect.set_alpha(0.6)
            else:
                rect.set_alpha(0.0)

        self._ax_margin.set_xlim(0, 1.1)
        self._ax_margin.set_ylim(self._omega_range[0], self._omega_range[1])

        total_power = float(np.sum(amp ** 2))
        n_crystal = int(np.sum(ms == 2))
        active_modes = int(np.sum(amp > np.percentile(amp, 50)))
        self.ax.set_title(
            f'spectrogram | pwr={total_power:.1f} | {active_modes}/{M} active | {n_crystal} crystal',
            fontsize=7, pad=2, color='#ddd',
        )
