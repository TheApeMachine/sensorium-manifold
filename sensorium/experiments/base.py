from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sensorium.projectors import (
    PipelineProjector, 
    LaTeXTableProjector, 
    FigureProjector, 
    ConsoleProjector, 
    TopTransitionsProjector, 
    TableConfig, 
    FigureConfig,
)

class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
        reportable: list[str] = [],
    ):
        self.experiment_name = experiment_name
        self.profile = profile
        self.dashboard = dashboard

        self.projector = PipelineProjector(
            # Console output for real-time feedback
            ConsoleProjector(
                fields=reportable,
                format="table",
            ),
            TopTransitionsProjector(),
            # LaTeX table - columns match InferenceObserver field names
            LaTeXTableProjector(
                TableConfig(
                    name=f"{experiment_name}_summary",
                    columns=reportable,
                    caption=f"{experiment_name} metrics",
                    label="tab:collision",
                    precision=3,
                ),
                output_dir=Path("paper/tables"),
            ),
            # Figure - x/y fields match InferenceObserver field names
            FigureProjector(
                FigureConfig(
                    name=f"{experiment_name}_metrics",
                    chart_type="line",
                    x="collision_rate",
                    y=["compression_ratio", "spatial_clustering", "entropy"],
                    title=f"{experiment_name} Metrics",
                    xlabel="Collision Rate",
                    ylabel="Metric Value",
                    grid=True,
                ),
                output_dir=Path("paper/figures"),
            ),
        )

    def close_dashboard(self):
        """Stop recording and close the dashboard."""
        if self._dashboard_instance is not None:
            self._dashboard_instance.stop_recording()
            self._dashboard_instance.close()
            self._dashboard_instance = None
            print(f"[dashboard] Video saved to {self.video_path}")

    @abstractmethod
    def observe(self, state: dict):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
