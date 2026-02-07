from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist


class Animation:
    """Thin wrapper around `matplotlib.animation.FuncAnimation`."""

    def __init__(
        self,
        fig,
        animate_frame: Callable[[int], Iterable[Artist]],
        *,
        interval_ms: int = 50,
    ) -> None:
        self.fig = fig
        self.animate_frame = animate_frame
        self.animation: FuncAnimation | None = FuncAnimation(
            self.fig,
            self.animate_frame,
            interval=int(interval_ms),
            blit=False,  # 3D doesn't support blitting reliably
            cache_frame_data=False,
        )

    def start(self) -> None:
        if self.animation is None:
            return
        es = getattr(self.animation, "event_source", None)
        if es is not None:
            es.start()

    def stop(self) -> None:
        if self.animation is None:
            return
        es = getattr(self.animation, "event_source", None)
        if es is not None:
            es.stop()

    def close(self) -> None:
        self.stop()
        if self.animation is not None:
            try:
                setattr(self.animation, "_draw_was_started", True)
            except Exception:
                pass
        self.animation = None
        self.animate_frame = lambda _frame_num: []

    def step(self) -> None:
        if self.animation is None:
            return
        step_fn = getattr(self.animation, "_step", None)
        if callable(step_fn):
            step_fn()
