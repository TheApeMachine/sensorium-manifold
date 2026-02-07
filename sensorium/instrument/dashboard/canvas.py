from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sensorium.instrument.dashboard.threed import ThreeD
from sensorium.instrument.dashboard.phase import PhasePlot
from sensorium.instrument.dashboard.spectrogram import SpectrogramPlot
from sensorium.instrument.dashboard.crystals import CrystalContentPlot
from sensorium.instrument.dashboard.settling import SettlingPlot


@dataclass(frozen=True)
class CanvasAxes:
    ax3d: object
    ax_phase: Optional[object] = None
    ax_spectrogram: Optional[object] = None
    ax_crystals: Optional[object] = None
    ax_settling: Optional[object] = None


class Canvas:
    """Two-column dashboard: 3D (left 50%) | analysis panels (right 50%)."""

    def __init__(self, grid_size: tuple[int, int, int], datafn) -> None:
        self.fig = plt.figure(figsize=(22, 11))

        # Squeeze margins to near zero
        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.97, bottom=0.03,
            wspace=0.06, hspace=0.0,
        )

        gs_main = gridspec.GridSpec(
            1, 2, figure=self.fig,
            width_ratios=[50, 50],
            wspace=0.06,
        )

        # Left: 3D full height
        self.ax3d = self.fig.add_subplot(gs_main[0, 0], projection="3d")

        # Right: 3 rows.
        # Row 1 is split into 2 columns: Psi phase | Crystallized modes.
        # Row 2/3 span full width: spectrogram, settling diagnostics.
        gs_right = gs_main[0, 1].subgridspec(
            3, 1,
            height_ratios=[30, 42, 20],
            # Extra vertical breathing room so spectrogram title/ticks do not collide
            # with Psi phase tick labels above.
            hspace=0.18,
        )

        gs_top = gs_right[0].subgridspec(
            1, 2,
            width_ratios=[58, 42],
            wspace=0.08,
        )
        self.ax_phase = self.fig.add_subplot(gs_top[0, 0])
        self.ax_crystals = self.fig.add_subplot(gs_top[0, 1])
        self.ax_spectrogram = self.fig.add_subplot(gs_right[1, 0])
        self.ax_settling = self.fig.add_subplot(gs_right[2, 0])

        # # Apply dark theme to all 2D axes
        # for ax in (self.ax_phase, self.ax_spectrogram, self.ax_crystals, self.ax_settling):
        #     ax.set_facecolor("#141424")
        #     ax.tick_params(colors="#aaa", labelsize=6)
        #     for spine in ax.spines.values():
        #         spine.set_color("#333")

        self.datafn = datafn
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

    def update(self, state: dict) -> None:
        for plot in self.plots.values():
            plot.update(state)
