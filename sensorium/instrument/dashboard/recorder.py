from __future__ import annotations

from pathlib import Path
from typing import Optional

from matplotlib.animation import FFMpegWriter


class Recorder:
    """FFmpeg-backed video recorder for a Matplotlib figure."""

    def __init__(self, fig) -> None:
        self.fig = fig
        self.path: Optional[Path] = None
        self._writer: Optional[FFMpegWriter] = None
        self._recording: bool = False

    def start(self, path: Path, *, fps: int = 30) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = FFMpegWriter(fps=int(fps))
        self._writer.setup(self.fig, str(self.path), dpi=getattr(self.fig, "dpi", 100))
        self._recording = True

    def stop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.finish()
            finally:
                self._writer = None
        self._recording = False

    def grab_frame(self) -> None:
        if not self._recording or self._writer is None:
            return
        self._writer.grab_frame()
