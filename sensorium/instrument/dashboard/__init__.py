"""Dashboard visualization components for the Sensorium Manifold.

Active panels (used by Canvas):
- Canvas: Main 2-column dashboard layout
- ThreeD: 3D particle visualization with phase/temperature coloring
- PhasePlot: Argand diagram + coherence ring + energy sparklines
- SpectrogramPlot: Rolling spectrogram with amplitude margin bars
- CrystalContentPlot: Decoded crystal byte content + compact metrics
- SettlingPlot: Convergence diagnostics (||ΔΨ||, CFL, Kuramoto R)

Utilities:
- Animation: Matplotlib animation wrapper
- Recorder: Frame capture and video export

Legacy panels (still importable but not instantiated by Canvas):
- EnergyPlot, InfoPlot, CoherencePlot, CouplingPlot, WavesPlot, SigmaPlot
"""

from sensorium.instrument.dashboard.canvas import Canvas, CanvasAxes
from sensorium.instrument.dashboard.animation import Animation
from sensorium.instrument.dashboard.session import DashboardSession

# Active visualizations
from sensorium.instrument.dashboard.threed import ThreeD
from sensorium.instrument.dashboard.phase import PhasePlot
from sensorium.instrument.dashboard.spectrogram import SpectrogramPlot
from sensorium.instrument.dashboard.crystals import CrystalContentPlot
from sensorium.instrument.dashboard.settling import SettlingPlot

__all__ = [
    # Main
    "Canvas",
    "CanvasAxes",
    "Animation",
    # Active panels
    "ThreeD",
    "PhasePlot",
    "SpectrogramPlot",
    "CrystalContentPlot",
    "SettlingPlot",
    "DashboardSession",
]
