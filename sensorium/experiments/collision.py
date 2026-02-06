"""Collision/compression scaling experiment (analysis mode).

This experiment is intentionally backend-agnostic so it can run on CPU-only
environments while still producing reviewer-relevant scaling curves for:
- key collisions/compression
- map-vs-path fold metrics
- SQL-based inference summaries
"""

from __future__ import annotations

import time
from pathlib import Path

from sensorium.dataset import SyntheticConfig, SyntheticDataset, SyntheticPattern
from sensorium.experiments.base import Experiment
from sensorium.experiments.state_builder import StateBuildConfig, build_observation_state
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import KeySpec, MapPathMetrics, ParticleCount, TokenDistributionMetrics, WaveFieldMetrics
from sensorium.observers.sql import SQLObserver, SQLObserverConfig
from sensorium.projectors import ConsoleProjector, LaTeXTableProjector, PipelineProjector, TableConfig


class CollisionExperiment(Experiment):
    """Run collision/compression sweeps at non-trivial token counts."""

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=[
                "scenario",
                "n_tokens",
                "num_units",
                "unit_length",
                "collision_rate",
                "compression_ratio",
                "collision_rate_observed",
                "n_unique_tokens",
                "unique_keys",
                "key_collision_rate",
                "fold_top1",
                "fold_pr",
                "transition_top1_prob",
                "transition_unique_edges",
                "mode_participation",
                "mode_entropy",
                "build_ms",
                "observe_ms",
                "sql_n_particles",
                "sql_n_unique_tokens",
                "sql_n_transition_edges",
                "sql_total_mass",
            ],
        )

        self.collision_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.scales = [
            {"name": "scale_032k", "num_units": 64, "unit_length": 512},
            {"name": "scale_131k", "num_units": 128, "unit_length": 1024},
        ]
        self.state_config = StateBuildConfig(grid_size=(64, 64, 64), mode_bins=512)

        self.inference = InferenceObserver(
            [
                ParticleCount(),
                TokenDistributionMetrics(),
                MapPathMetrics(key=KeySpec("sequence_byte"), topk=20),
                WaveFieldMetrics(),
                SQLObserver(
                    """
                    SELECT
                        n_particles,
                        n_unique_tokens,
                        n_transition_edges,
                        total_mass
                    FROM simulation;
                    """,
                    config=SQLObserverConfig(row_limit=1),
                ),
            ]
        )

        self.projector = PipelineProjector(
            ConsoleProjector(fields=self.reportable, format="table"),
            LaTeXTableProjector(
                TableConfig(
                    name="collision_summary",
                    columns=self.reportable,
                    caption="Collision/compression scaling summary (analysis mode)",
                    label="tab:collision",
                    precision=4,
                ),
                output_dir=Path("paper/tables"),
            ),
        )

    def run(self):
        for scale in self.scales:
            for rate in self.collision_rates:
                dataset = SyntheticDataset(
                    SyntheticConfig(
                        pattern=SyntheticPattern.COLLISION,
                        num_units=int(scale["num_units"]),
                        unit_length=int(scale["unit_length"]),
                        collision_rate=float(rate),
                        seed=42,
                    )
                )

                t0 = time.perf_counter()
                state = build_observation_state(dataset.generate(), config=self.state_config)
                build_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                self.inference.observe(
                    state,
                    scenario=str(scale["name"]),
                    run_name=f"{scale['name']}_r{rate:.1f}",
                    n_tokens=int(state["token_ids"].numel()),
                    num_units=int(scale["num_units"]),
                    unit_length=int(scale["unit_length"]),
                    collision_rate=float(rate),
                    build_ms=float(build_ms),
                    observe_ms=0.0,  # set below after observer execution
                )
                observe_ms = (time.perf_counter() - t1) * 1000.0
                self.inference.results[-1]["observe_ms"] = float(observe_ms)

        return self.project()

    def observe(self, state: dict) -> dict:
        return self.inference.observe(state)

    def project(self) -> dict:
        return self.projector.project(self.inference)
