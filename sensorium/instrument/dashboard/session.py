from __future__ import annotations

import os
import time
from pathlib import Path


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

        from sensorium.instrument.dashboard.animation import Animation
        from sensorium.instrument.dashboard.canvas import Canvas
        from sensorium.instrument.dashboard.recorder import Recorder

        gx, gy, gz = grid_size
        self.grid_size: tuple[int, int, int] = (int(gx), int(gy), int(gz))
        self.fps = int(max(1, fps))
        backend_name = str(plt.get_backend()).lower()
        backend_is_interactive = "agg" not in backend_name
        self.show = bool(show and backend_is_interactive)

        self._plt = plt
        self.canvas = Canvas(grid_size=self.grid_size, datafn=lambda: None)
        self.recorder = Recorder(self.canvas.fig)

        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.recorder.start(self.video_path, fps=self.fps)

        self._frame_period = 1.0 / float(self.fps)
        self._last_frame_t = 0.0
        self._last_event_pump_t = 0.0
        self._event_pump_period = min(0.02, self._frame_period)
        self._pending_state: dict | None = None

        self.animation = None
        if self.show:
            self.animation = Animation(
                self.canvas.fig,
                self._animate_frame,
                interval_ms=max(1, int(round(1000.0 / float(self.fps)))),
            )
        self._manual_render = not self.show

        if self.show:
            try:
                plt.ion()
                plt.show(block=False)
                if self.animation is not None:
                    self.animation.start()
            except Exception:
                # Headless / non-interactive environments can still record.
                pass

    @staticmethod
    def from_env(
        *, grid_size: tuple[int, int, int], video_path: Path
    ) -> DashboardSession:
        fps_s = os.environ.get("THERMO_MANIFOLD_DASHBOARD_FPS", "30")
        try:
            fps = int(fps_s)
        except Exception:
            fps = 30
        return DashboardSession(
            grid_size=grid_size, video_path=video_path, fps=fps, show=True
        )

    def update(self, state: dict) -> None:
        # Keep only the latest state from manifold; render loop ingests on frame cadence.
        self._pending_state = state

        # In headless mode, drive rendering directly with a lightweight frame cadence.
        if self._manual_render:
            self._animate_frame(0)
            return

        # Pump GUI events so FuncAnimation callbacks run.
        now = time.perf_counter()
        if (now - self._last_event_pump_t) < self._event_pump_period:
            return
        self._last_event_pump_t = now
        try:
            if self.animation is not None:
                self.animation.step()
                return
            # Fallback path for unexpected animation setup failures.
            self._plt.pause(0.001)
        except Exception:
            pass

    def _animate_frame(self, _frame_num: int) -> list:
        now = time.perf_counter()
        if (now - self._last_frame_t) < self._frame_period:
            return []

        state = self._pending_state
        if state is None:
            return []

        try:
            self.canvas.ingest(state)
            self.canvas.render()
        except Exception:
            return []
        self._last_frame_t = now
        self._pending_state = None
        self.recorder.grab_frame()
        return []

    def stop_recording(self) -> None:
        try:
            if self.animation is not None:
                self.animation.stop()
        except Exception:
            pass
        self.recorder.stop()

    def close(self) -> None:
        try:
            torch_mod = __import__("torch")
            if torch_mod.backends.mps.is_available():
                torch_mod.mps.synchronize()
        except Exception:
            pass
        try:
            if self.animation is not None:
                self.animation.close()
        except Exception:
            pass
        try:
            self._plt.close(self.canvas.fig)
        except Exception:
            pass
