from __future__ import annotations

from typing import Callable, Optional

from matplotlib.animation import FuncAnimation


class Animation:
    """Thin wrapper around `matplotlib.animation.FuncAnimation`."""

    def __init__(self, fig, animate_frame: Callable[[int], object], *, interval_ms: int = 50) -> None:
        self.fig = fig
        self.animate_frame = animate_frame
        self.animation = FuncAnimation(
            self.fig,
            self.animate_frame,
            interval=int(interval_ms),
            blit=False,  # 3D doesn't support blitting reliably
            cache_frame_data=False,
        )

    def start(self) -> None:
        es = getattr(self.animation, "event_source", None)
        if es is not None:
            es.start()

    def stop(self) -> None:
        es = getattr(self.animation, "event_source", None)
        if es is not None:
            es.stop()

    def close(self) -> None:
        # Matplotlib animations do not always expose a close() API; stopping is enough.
        self.stop()