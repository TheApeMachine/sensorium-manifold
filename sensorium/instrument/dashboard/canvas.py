from __future__ import annotations

from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sensorium.instrument.dashboard.crystals import CrystalContentPlot
from sensorium.instrument.dashboard.phase import PhasePlot
from sensorium.instrument.dashboard.settling import SettlingPlot
from sensorium.instrument.dashboard.spectrogram import SpectrogramPlot
from sensorium.instrument.dashboard.threed import ThreeD


@dataclass(frozen=True)
class CanvasAxes:
    ax3d: object
    ax_phase: object | None = None
    ax_spectrogram: object | None = None
    ax_crystals: object | None = None
    ax_settling: object | None = None


class Canvas:
    """Dashboard layout with 3D + spectrogram on the left and analysis on the right."""

    def __init__(self, grid_size: tuple[int, int, int], datafn) -> None:
        # Keep a practical default window size so panels stay on-screen on laptops.
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(
            left=0.04,
            right=0.98,
            top=0.96,
            bottom=0.06,
            wspace=0.12,
            hspace=0.16,
        )

        gs_main = gridspec.GridSpec(
            2,
            2,
            figure=self.fig,
            width_ratios=[64, 36],
            height_ratios=[74, 26],
            wspace=0.12,
            hspace=0.16,
        )

        # Left column: 3D on top, spectrogram below.
        self.ax3d = self.fig.add_subplot(gs_main[0, 0], projection="3d")
        self.ax_spectrogram = self.fig.add_subplot(gs_main[1, 0])

        # Right column spans both rows:
        # top split: Psi phase | Crystallized modes
        # bottom: Settling diagnostics
        gs_right = gs_main[:, 1].subgridspec(
            2,
            1,
            height_ratios=[64, 36],
            hspace=0.24,
        )
        gs_top = gs_right[0].subgridspec(
            1,
            2,
            width_ratios=[58, 42],
            wspace=0.14,
        )
        self.ax_phase = self.fig.add_subplot(gs_top[0, 0])
        self.ax_crystals = self.fig.add_subplot(gs_top[0, 1])
        self.ax_settling = self.fig.add_subplot(gs_right[1, 0])

        self.datafn = datafn
        self._render_array_keys = (
            "positions",
            "velocities",
            "phase",
            "energy_osc",
            "heats",
            "masses",
            "psi_real",
            "psi_imag",
            "mode_state",
            "omega_lattice",
            "psi_amplitude",
            "mode_anchor_idx",
            "mode_anchor_weight",
            "token_ids",
            "sequence_indices",
            "energies",
            "gravity_potential",
        )
        self._render_scalar_keys = (
            "step",
            "dt",
            "c_v",
            "cfl_max_rate",
            "spatial_sigma",
        )
        self.plots = {
            "three": ThreeD(grid_size=grid_size, ax=self.ax3d),
            "phase": PhasePlot(ax=self.ax_phase),
            "spectrogram": SpectrogramPlot(ax=self.ax_spectrogram),
            "crystals": CrystalContentPlot(ax=self.ax_crystals),
            "settling": SettlingPlot(ax=self.ax_settling),
        }

        self.fig.canvas.draw_idle()

    def animate_frame(self, frame_num: int) -> list:
        self.datafn()
        return []

    def init(self) -> CanvasAxes:
        return CanvasAxes(
            ax3d=self.ax3d,
            ax_phase=self.ax_phase,
            ax_spectrogram=self.ax_spectrogram,
            ax_crystals=self.ax_crystals,
            ax_settling=self.ax_settling,
        )

    def ingest(self, state: dict) -> None:
        render_state = self._prepare_render_state(state)
        for plot in self.plots.values():
            ingest_fn = getattr(plot, "ingest", None)
            if callable(ingest_fn):
                ingest_fn(render_state)

    def render(self) -> None:
        for plot in self.plots.values():
            render_fn = getattr(plot, "render", None)
            if callable(render_fn):
                render_fn()

    def update(self, state: dict) -> None:
        """Compatibility path: ingest + render in one call."""
        self.ingest(state)

    def _prepare_render_state(self, state: dict) -> dict:
        """Prepare a lightweight dashboard state view for panel ingest methods."""
        out = dict(state)
        for key in self._render_scalar_keys:
            v = out.get(key)
            if v is not None and hasattr(v, "item"):
                try:
                    out[key] = float(v.item())
                except Exception:
                    pass
        return out
