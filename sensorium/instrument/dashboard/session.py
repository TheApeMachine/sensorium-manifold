from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


class DashboardSession:
    """Live dashboard + video recording for experiments.

    This is intentionally minimal:
    - `update(state)` renders plots into a Matplotlib figure
    - frames are captured via FFmpegWriter (mp4)
    - when an interactive backend is available, the window is shown
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int, int],
        video_path: Path,
        fps: int = 30,
        show: bool = True,
    ) -> None:
        # Import pyplot lazily so CLI can set backend first.
        import matplotlib.pyplot as plt

        from sensorium.instrument.dashboard.canvas import Canvas
        from sensorium.instrument.dashboard.recorder import Recorder

        gx, gy, gz = grid_size
        self.grid_size: tuple[int, int, int] = (int(gx), int(gy), int(gz))
        self.fps = int(max(1, fps))
        self.show = bool(show)

        self._plt = plt
        self.canvas = Canvas(grid_size=self.grid_size, datafn=lambda: None)
        self.recorder = Recorder(self.canvas.fig)

        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.recorder.start(self.video_path, fps=self.fps)

        self._frame_period = 1.0 / float(self.fps)
        self._last_frame_t = 0.0

        if self.show:
            try:
                plt.ion()
                plt.show(block=False)
            except Exception:
                # Headless / non-interactive environments can still record.
                pass

    @staticmethod
    def from_env(
        *, grid_size: tuple[int, int, int], video_path: Path
    ) -> "DashboardSession":
        fps_s = os.environ.get("THERMO_MANIFOLD_DASHBOARD_FPS", "30")
        try:
            fps = int(fps_s)
        except Exception:
            fps = 30
        return DashboardSession(
            grid_size=grid_size, video_path=video_path, fps=fps, show=True
        )

    def update(self, state: dict) -> None:
        # Always update plot state, but throttle frame grabs.
        try:
            self.canvas.update(state)
        except Exception:
            return

        now = time.perf_counter()
        if (now - self._last_frame_t) < self._frame_period:
            return
        self._last_frame_t = now

        try:
            self.canvas.fig.canvas.draw_idle()
            self.canvas.fig.canvas.flush_events()
            # Yield to GUI; safe no-op on non-interactive backends.
            self._plt.pause(0.001)
        except Exception:
            pass
        self.recorder.grab_frame()

    def stop_recording(self) -> None:
        self.recorder.stop()

    def close(self) -> None:
        try:
            self._plt.close(self.canvas.fig)
        except Exception:
            pass
