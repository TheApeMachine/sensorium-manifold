"""Wave-trie / map-vs-path branching experiment (analysis mode)."""

from __future__ import annotations

from pathlib import Path

from sensorium.dataset import SyntheticConfig, SyntheticDataset, SyntheticPattern
from sensorium.experiments.base import Experiment
from sensorium.experiments.state_builder import StateBuildConfig, build_observation_state
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import KeySpec, MapPathMetrics, TokenDistributionMetrics, WaveFieldMetrics
from sensorium.observers.sql import SQLObserver, SQLObserverConfig
from sensorium.projectors import ConsoleProjector, LaTeXTableProjector, PipelineProjector, TableConfig, TopTransitionsProjector


class WaveTrieExperiment(Experiment):
    """Measure branch structure and fold statistics for shared-prefix datasets."""

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=[
                "scenario",
                "n_tokens",
                "n_samples",
                "compression_ratio",
                "collision_rate_observed",
                "unique_keys",
                "key_collision_rate",
                "transition_edges",
                "transition_unique_edges",
                "transition_top1_prob",
                "transition_entropy",
                "fold_top1",
                "fold_pr",
                "fold_entropy",
                "mode_participation",
                "mode_entropy",
                "sql_n_branch_edges",
                "sql_max_branch_count",
                "sql_mean_branch_count",
            ],
        )

        self.state_config = StateBuildConfig(grid_size=(64, 64, 64), mode_bins=512)
        self.scenarios = [
            {
                "name": "abc_abd_abe",
                "patterns": ["ABC", "ABD", "ABE"],
                "counts": {"ABC": 128, "ABD": 128, "ABE": 128},
            },
            {
                "name": "ab_repeat_branch",
                "patterns": ["ABAB", "ABAC", "ABAD", "ABAE"],
                "counts": {"ABAB": 96, "ABAC": 96, "ABAD": 96, "ABAE": 96},
            },
        ]

        self.inference = InferenceObserver(
            [
                TokenDistributionMetrics(),
                MapPathMetrics(key=KeySpec("sequence_byte"), topk=20),
                WaveFieldMetrics(),
                SQLObserver(
                    """
                    SELECT
                        COUNT(*) AS n_branch_edges,
                        MAX(edge_count) AS max_branch_count,
                        AVG(edge_count) AS mean_branch_count
                    FROM transitions;
                    """,
                    config=SQLObserverConfig(row_limit=1),
                ),
            ]
        )

        self.projector = PipelineProjector(
            ConsoleProjector(fields=self.reportable, format="table"),
            LaTeXTableProjector(
                TableConfig(
                    name="wave_trie_summary",
                    columns=self.reportable,
                    caption="Wave-trie branch/fold summary (analysis mode)",
                    label="tab:wave_trie",
                    precision=4,
                ),
                output_dir=Path("paper/tables"),
            ),
            TopTransitionsProjector(),
        )

    def run(self):
        for item in self.scenarios:
            dataset = SyntheticDataset(
                SyntheticConfig(
                    pattern=SyntheticPattern.TEXT_PREFIX,
                    text_patterns=list(item["patterns"]),
                    pattern_counts=dict(item["counts"]),
                    seed=42,
                )
            )
            state = build_observation_state(dataset.generate(), config=self.state_config)

            self.inference.observe(
                state,
                scenario=str(item["name"]),
                run_name=str(item["name"]),
                n_tokens=int(state["token_ids"].numel()),
            )

        return self.project()

    def observe(self, state: dict) -> dict:
        return self.inference.observe(state)

    def project(self) -> dict:
        return self.projector.project(self.inference)
