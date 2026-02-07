from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from sensorium.projectors import (
    PipelineProjector,
    LaTeXTableProjector,
    FigureProjector,
    ConsoleProjector,
    TopTransitionsProjector,
    TableConfig,
    FigureConfig,
)


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
        reportable: Sequence[str] | None = None,
    ):
        self.experiment_name = slugify(experiment_name)
        self.profile = profile
        self.dashboard = dashboard
        self.reportable = list(reportable) if reportable else []
        self.repo_root = Path(__file__).resolve().parents[2]
        self._dashboard_instance = None
        self._artifact_dir = self.repo_root / "artifacts" / self.experiment_name
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        # Default: keep videos next to paper assets.
        self.video_path = str(
            self.repo_root
            / "paper"
            / "videos"
            / f"{self.experiment_name}_dashboard.mp4"
        )

        self.projector = PipelineProjector(
            # Console output for real-time feedback
            ConsoleProjector(
                fields=self.reportable,
                format="table",
            ),
            TopTransitionsProjector(),
            # LaTeX table - columns match InferenceObserver field names
            LaTeXTableProjector(
                TableConfig(
                    name=f"{self.experiment_name}_summary",
                    columns=self.reportable,
                    caption=f"{experiment_name} metrics",
                    label=f"tab:{self.experiment_name}",
                    precision=3,
                ),
                output_dir=Path("paper/tables"),
            ),
        )

    def _slug(self) -> str:
        return "".join(
            ch if ch.isalnum() else "_" for ch in self.experiment_name.lower()
        ).strip("_")

    def artifact_path(self, *parts: str) -> Path:
        """Resolve an artifact path and create parent directories."""
        if parts and parts[0] in {"tables", "figures"}:
            base = self.repo_root / "paper"
        else:
            base = self._artifact_dir
        path = base.joinpath(*parts) if parts else base
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def close_dashboard(self):
        """Stop recording and close the dashboard."""
        if self._dashboard_instance is not None:
            self._dashboard_instance.stop_recording()
            self._dashboard_instance.close()
            self._dashboard_instance = None
            print(f"[dashboard] Video saved to {self.video_path}")

    def start_dashboard(
        self, *, grid_size: tuple[int, int, int], run_name: Optional[str] = None
    ) -> None:
        """Start live dashboard + mp4 recording for this run."""
        from sensorium.instrument.dashboard import DashboardSession

        # Restart if already running.
        self.close_dashboard()

        name = (run_name or self.experiment_name).strip() or self.experiment_name
        video_path = (
            self.repo_root / "paper" / "videos" / f"{slugify(name)}_dashboard.mp4"
        )
        self.video_path = str(video_path)
        gx, gy, gz = grid_size
        self._dashboard_instance = DashboardSession.from_env(
            grid_size=(int(gx), int(gy), int(gz)),
            video_path=video_path,
        )

    def dashboard_update(self, state: dict) -> None:
        if self._dashboard_instance is None:
            return
        self._dashboard_instance.update(state)

    @abstractmethod
    def observe(self, state: dict):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
