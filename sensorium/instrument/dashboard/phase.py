"""Combined Phase Space + Coherence + Energy visualization.

Argand diagram with Kuramoto R ring, mode-state arcs, and energy sparklines.
Dark theme -- all text/lines use visible colors on dark backgrounds.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class PhasePlot:
    """Combined Argand diagram with coherence ring and energy sparklines."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self._max_trail = 12
        self._trail_history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self._max_trail)

        # Pre-create artists
        self._trail_scatters: list[object] = []
        self._scatter_current = ax.scatter(
            [], [], s=[], c=[], alpha=0.85,
            edgecolors='#222', linewidths=0.3, zorder=10,
        )
        self._scatter_mean = ax.scatter(
            [], [], s=100, c='#e74c3c', marker='x', linewidths=2, zorder=20,
        )
        self._order_arrow = ax.annotate(
            '', xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2),
        )

        # Reference circles + cardinal lines -- darker colors for dark bg
        self._ref_circles: list[object] = []
        self._ref_lines: list[object] = []

        # Kuramoto R ring
        (self._r_ring,) = ax.plot([], [], lw=2.5, alpha=0.8, color='#9b59b6', zorder=5)

        # Mode-state arc segments
        (self._arc_neutral,) = ax.plot([], [], lw=4, alpha=0.5, color='#7f8c8d', zorder=4, solid_capstyle='round')
        (self._arc_stable,)  = ax.plot([], [], lw=4, alpha=0.7, color='#3498db', zorder=4, solid_capstyle='round')
        (self._arc_crystal,) = ax.plot([], [], lw=4, alpha=0.9, color='#f39c12', zorder=4, solid_capstyle='round')

        self._initialized = False
        self._theta: Optional[np.ndarray] = None

        self._cmap_state = {0: '#7f8c8d', 1: '#3498db', 2: '#f39c12'}

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#7f8c8d', markersize=5, label='neutral'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#3498db', markersize=5, label='stable'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#f39c12', markersize=5, label='crystal'),
        ]
        ax.legend(handles=handles, fontsize=5, loc='upper left', framealpha=0.4,
                  facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')
        ax.set_xlabel('Re(Psi)', fontsize=7, color='#aaa')
        ax.set_ylabel('Im(Psi)', fontsize=7, color='#aaa')
        ax.tick_params(labelsize=5, colors='#aaa')

        # Energy sparklines inset
        self._energy_history: dict[str, deque] = {
            "step": deque(maxlen=200),
            "emode": deque(maxlen=200),
            "eheat": deque(maxlen=200),
            "ekin":  deque(maxlen=200),
        }
        self._ax_energy = ax.inset_axes([0.72, 0.03, 0.26, 0.26])
        self._ax_energy.set_facecolor("#1a1a2e")
        self._ax_energy.patch.set_alpha(0.85)
        self._ax_energy.tick_params(labelsize=4, colors='#999', length=2)
        self._ax_energy.set_ylabel('E', fontsize=5, color='#999')
        for spine in self._ax_energy.spines.values():
            spine.set_color('#333')
        (self._ln_emode,) = self._ax_energy.plot([], [], color='#2980b9', lw=1.0, label='mode')
        (self._ln_eheat,) = self._ax_energy.plot([], [], color='#c0392b', lw=1.0, label='heat')
        (self._ln_ekin,)  = self._ax_energy.plot([], [], color='#7f8c8d', lw=1.0, label='kin')
        (self._ln_etot,)  = self._ax_energy.plot([], [], color='#8e44ad', lw=1.0, ls='--', label='tot')
        self._ax_energy.legend(fontsize=4, loc='upper left', framealpha=0.4,
                               facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc', ncol=2)
        self._energy_step = 0

    def update(self, state: dict) -> None:
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v

        self._update_energy(state)

        psi_real = get("psi_real")
        psi_imag = get("psi_imag")
        mode_state_arr = get("mode_state")

        if psi_real is None or psi_imag is None:
            self.ax.set_title("Psi phase space: awaiting data", fontsize=8, color='#ccc')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self._scatter_current.set_offsets(np.zeros((0, 2), dtype=np.float64))
            return

        psi_r = psi_real.astype(np.float64)
        psi_i = psi_imag.astype(np.float64)
        ms = mode_state_arr.astype(np.int32) if mode_state_arr is not None else np.zeros(len(psi_r), dtype=np.int32)
        self._trail_history.append((psi_r.copy(), psi_i.copy(), ms.copy()))

        amp = np.sqrt(psi_r * psi_r + psi_i * psi_i)
        r_max = float(np.percentile(amp[amp > 0], 98)) * 1.3 if (len(amp) > 0 and np.any(amp > 0)) else 1.0
        r_max = max(r_max, 0.1)

        if not self._initialized:
            theta = np.linspace(0, 2 * np.pi, 200)
            for _ in range(4):
                (ln,) = self.ax.plot([], [], color='#444', lw=0.5, alpha=0.6, zorder=0)
                self._ref_circles.append(ln)
            for _ in range(4):
                (ln,) = self.ax.plot([], [], color='#444', lw=0.3, alpha=0.5, zorder=0)
                self._ref_lines.append(ln)
            for _ in range(self._max_trail - 1):
                sc = self.ax.scatter([], [], s=4, c='#666', alpha=0.15, edgecolors='none', zorder=1)
                self._trail_scatters.append(sc)
            self._theta = theta
            self._initialized = True

        theta = self._theta

        for i, frac in enumerate([0.25, 0.5, 0.75, 1.0]):
            rr = frac * r_max
            self._ref_circles[i].set_data(rr * np.cos(theta), rr * np.sin(theta))
        for i, angle in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):
            self._ref_lines[i].set_data([0, r_max * np.cos(angle)], [0, r_max * np.sin(angle)])

        # Kuramoto R ring
        phases = np.arctan2(psi_i, psi_r)
        order_complex = np.mean(np.exp(1j * phases))
        order_param = float(np.abs(order_complex))
        mean_phase = float(np.angle(order_complex))

        r_ring_radius = order_param * r_max
        self._r_ring.set_data(r_ring_radius * np.cos(theta), r_ring_radius * np.sin(theta))
        self._r_ring.set_color('#27ae60' if order_param > 0.7 else '#f39c12' if order_param > 0.4 else '#e74c3c')

        # Mode-state arcs
        M = len(ms)
        if M > 0:
            fracs = np.array([int(np.sum(ms == 0)), int(np.sum(ms == 1)), int(np.sum(ms == 2))], dtype=np.float64) / M
            angles = np.cumsum(np.concatenate([[0.0], fracs])) * 2 * np.pi
            r_arc = r_max * 1.08
            for arc_line, s, e in [
                (self._arc_neutral, angles[0], angles[1]),
                (self._arc_stable,  angles[1], angles[2]),
                (self._arc_crystal, angles[2], angles[3]),
            ]:
                if e > s + 1e-6:
                    t = np.linspace(s, e, max(int((e - s) / 0.05), 3))
                    arc_line.set_data(r_arc * np.cos(t), r_arc * np.sin(t))
                else:
                    arc_line.set_data([], [])
        else:
            for a in (self._arc_neutral, self._arc_stable, self._arc_crystal):
                a.set_data([], [])

        # Trails
        trail_len = len(self._trail_history)
        for si, sc in enumerate(self._trail_scatters):
            hist_index = trail_len - (self._max_trail - si)
            if hist_index < 0 or hist_index >= (trail_len - 1):
                sc.set_offsets(np.zeros((0, 2), dtype=np.float64))
                continue
            tr, ti_arr, _ = self._trail_history[hist_index]
            age = (trail_len - 1) - hist_index
            alpha = 0.08 + 0.15 * (1.0 - age / max(trail_len - 1, 1))
            size = 4 + 8 * (1.0 - age / max(trail_len - 1, 1))
            sc.set_offsets(np.column_stack([tr, ti_arr]))
            sc.set_alpha(alpha)
            sc.set_sizes(np.full((len(tr),), size, dtype=np.float64))

        # Current scatter
        colors = np.array([self._cmap_state.get(int(s), '#7f8c8d') for s in ms])
        sizes = np.clip(20 + 80 * (amp / (r_max / 1.3 + 1e-8)), 15, 150)
        self._scatter_current.set_offsets(np.column_stack([psi_r, psi_i]))
        self._scatter_current.set_sizes(sizes.astype(np.float64))
        self._scatter_current.set_facecolors(colors)

        self._scatter_mean.set_offsets(np.array([[float(np.mean(psi_r)), float(np.mean(psi_i))]], dtype=np.float64))
        self._order_arrow.xy = (order_param * r_max * np.cos(mean_phase),
                                order_param * r_max * np.sin(mean_phase))

        pad = r_max * 1.15
        self.ax.set_xlim(-pad, pad)
        self.ax.set_ylim(-pad, pad)
        self.ax.set_aspect('equal')

        n_stable = int(np.sum(ms == 1))
        n_crystal = int(np.sum(ms == 2))
        h = self._energy_history
        total_E = (h["emode"][-1] if h["emode"] else 0.0) + (h["eheat"][-1] if h["eheat"] else 0.0) + (h["ekin"][-1] if h["ekin"] else 0.0)
        self.ax.set_title(
            f'Psi phase | R={order_param:.2f} | {n_stable}s {n_crystal}c | E={total_E:.1f}',
            fontsize=8, pad=2, color='#ddd',
        )

    def _update_energy(self, state: dict) -> None:
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v

        self._energy_step += 1
        energy = get("energies", np.array([]))
        heat = get("heats", np.array([]))
        masses = get("masses")
        velocities = get("velocities", np.zeros((0, 3)))

        e_mode = float(energy.sum()) if len(energy) > 0 else 0.0
        eheat = float(heat.sum()) if len(heat) > 0 else 0.0
        m_eff = masses if masses is not None else energy
        v2 = (velocities ** 2).sum(axis=1) if len(velocities) > 0 else np.array([])
        ekin = float(0.5 * np.sum(m_eff * v2)) if len(v2) > 0 else 0.0

        h = self._energy_history
        h["step"].append(self._energy_step)
        h["emode"].append(e_mode)
        h["eheat"].append(eheat)
        h["ekin"].append(ekin)

        steps = list(h["step"])
        if len(steps) < 2:
            return

        emode_a = np.asarray(h["emode"], dtype=np.float64)
        eheat_a = np.asarray(h["eheat"], dtype=np.float64)
        ekin_a  = np.asarray(h["ekin"], dtype=np.float64)
        etot_a  = emode_a + eheat_a + ekin_a

        self._ln_emode.set_data(steps, emode_a)
        self._ln_eheat.set_data(steps, eheat_a)
        self._ln_ekin.set_data(steps, ekin_a)
        self._ln_etot.set_data(steps, etot_a)

        self._ax_energy.relim()
        self._ax_energy.autoscale_view()
