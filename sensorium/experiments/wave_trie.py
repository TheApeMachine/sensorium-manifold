"""Wave-trie / map-vs-path branching experiment on real manifold dynamics."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from sensorium.dataset import SyntheticConfig, SyntheticDataset, SyntheticPattern
from sensorium.experiments.base import Experiment
from sensorium.experiments.manifold_runner import (
    ManifoldRunConfig,
    run_stream_on_manifold,
)
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    KeySpec,
    MapPathMetrics,
    TokenDistributionMetrics,
    WaveFieldMetrics,
)
from sensorium.observers.sql import SQLObserver, SQLObserverConfig
from sensorium.projectors import (
    ConsoleProjector,
    LaTeXTableProjector,
    PipelineProjector,
    TableConfig,
    TopTransitionsProjector,
)


class WaveTrieExperiment(Experiment):
    """Measure branch structure and fold statistics for shared-prefix datasets."""

    def __init__(
        self, experiment_name: str, profile: bool = False, dashboard: bool = False
    ):
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
                "n_patterns",
                "families",
                "branches_per_family",
                "repeats_per_pattern",
                "expected_unique_edges",
                "expected_branch_nodes",
                "expected_max_out_degree",
                "transition_edges",
                "transition_unique_edges",
                "branch_edge_coverage",
                "out_degree_coverage",
                "transition_top1_prob",
                "transition_entropy",
                "fold_top1",
                "fold_pr",
                "fold_entropy",
                "mode_participation",
                "mode_entropy",
                "run_backend",
                "run_steps",
                "run_termination",
                "init_ms",
                "simulate_ms",
                "sql_n_branch_edges",
                "sql_max_branch_count",
                "sql_mean_branch_count",
                "sql_max_out_degree",
                "sql_mean_out_degree",
                "sql_n_branch_nodes",
            ],
        )

        allow_fallback = os.getenv(
            "SENSORIUM_ALLOW_ANALYSIS_FALLBACK", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.run_config = ManifoldRunConfig(
            grid_size=(64, 64, 64),
            max_steps=16,
            min_steps=4,
            allow_analysis_fallback=allow_fallback,
            analysis_mode_bins=512,
        )
        self.scenarios: List[Dict[str, Any]] = self._build_scenarios()

        self.inference = InferenceObserver(
            [
                TokenDistributionMetrics().observe,
                MapPathMetrics(key=KeySpec("sequence_byte"), topk=20).observe,
                WaveFieldMetrics().observe,
                SQLObserver(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM transitions) AS n_branch_edges,
                        (SELECT MAX(edge_count) FROM transitions) AS max_branch_count,
                        (SELECT AVG(edge_count) FROM transitions) AS mean_branch_count,
                        (
                            SELECT COALESCE(MAX(out_degree), 0)
                            FROM (
                                SELECT src_token_id, COUNT(DISTINCT dst_token_id) AS out_degree
                                FROM transitions
                                GROUP BY src_token_id
                            )
                        ) AS max_out_degree,
                        (
                            SELECT COALESCE(AVG(out_degree), 0.0)
                            FROM (
                                SELECT src_token_id, COUNT(DISTINCT dst_token_id) AS out_degree
                                FROM transitions
                                GROUP BY src_token_id
                            )
                        ) AS mean_out_degree,
                        (
                            SELECT COALESCE(SUM(CASE WHEN out_degree > 1 THEN 1 ELSE 0 END), 0)
                            FROM (
                                SELECT src_token_id, COUNT(DISTINCT dst_token_id) AS out_degree
                                FROM transitions
                                GROUP BY src_token_id
                            )
                        ) AS n_branch_nodes;
                    """,
                    config=SQLObserverConfig(row_limit=1),
                ).observe,
            ]
        )

        self.projector = PipelineProjector(
            ConsoleProjector(fields=self.reportable, format="table"),
            LaTeXTableProjector(
                TableConfig(
                    name="wave_trie_summary",
                    columns=self.reportable,
                    caption="Wave-trie branch/fold summary (real manifold)",
                    label="tab:wave_trie",
                    precision=4,
                ),
                output_dir=Path("paper/tables"),
            ),
            TopTransitionsProjector(),
        )

    def _branch_code(self, value: int, width: int, alphabet: str) -> str:
        x = int(max(0, value))
        base = int(max(2, len(alphabet)))
        out: List[str] = []
        for _ in range(width):
            out.append(alphabet[x % base])
            x //= base
        # Keep least-significant symbol first so early branch positions vary the most.
        return "".join(out)

    def _make_patterns(
        self, *, families: int, branches_per_family: int, tail_width: int = 3
    ) -> List[str]:
        alphabet = "ABCDEFGH"  # 8 symbols; encourages multi-level branching
        min_width = int(
            max(1, math.ceil(math.log(max(1, int(branches_per_family)), len(alphabet))))
        )
        width = int(max(int(tail_width), min_width))
        patterns: List[str] = []
        for fam in range(int(families)):
            prefix = f"F{fam:02d}|ROOT|"
            for branch in range(int(branches_per_family)):
                # Multi-position branch code (not a single suffix hotspot).
                code = self._branch_code(branch, width, alphabet)
                patterns.append(f"{prefix}{code}|END")
        return patterns

    def _expected_transition_stats(self, patterns: List[str]) -> Dict[str, int]:
        edges: set[tuple[int, int]] = set()
        out_adj: dict[int, set[int]] = {}
        for pattern in patterns:
            data = pattern.encode("utf-8")
            if len(data) < 2:
                continue
            for i in range(len(data) - 1):
                src = ((int(i) & 0xFFFFFFFF) << 8) | (int(data[i]) & 0xFF)
                dst = ((int(i + 1) & 0xFFFFFFFF) << 8) | (int(data[i + 1]) & 0xFF)
                edges.add((src, dst))
                out_adj.setdefault(src, set()).add(dst)

        max_out = max((len(v) for v in out_adj.values()), default=0)
        n_branch_nodes = sum(1 for v in out_adj.values() if len(v) > 1)
        return {
            "expected_unique_edges": int(len(edges)),
            "expected_max_out_degree": int(max_out),
            "expected_branch_nodes": int(n_branch_nodes),
        }

    def _build_scenarios(self) -> List[Dict[str, Any]]:
        defs = [
            {
                "name": "concurrent_3",
                "families": 1,
                "branches_per_family": 3,
                "repeats": 256,
            },
            {
                "name": "concurrent_64",
                "families": 4,
                "branches_per_family": 16,
                "repeats": 64,
            },
            {
                "name": "concurrent_256",
                "families": 8,
                "branches_per_family": 32,
                "repeats": 24,
            },
        ]
        scenarios: List[Dict[str, Any]] = []
        for item in defs:
            patterns = self._make_patterns(
                families=int(item["families"]),
                branches_per_family=int(item["branches_per_family"]),
                tail_width=3,
            )
            counts = {p: int(item["repeats"]) for p in patterns}
            expected = self._expected_transition_stats(patterns)
            scenarios.append(
                {
                    "name": str(item["name"]),
                    "patterns": patterns,
                    "counts": counts,
                    "families": int(item["families"]),
                    "branches_per_family": int(item["branches_per_family"]),
                    "repeats": int(item["repeats"]),
                    **expected,
                }
            )
        return scenarios

    def run(self):
        for item in self.scenarios:
            patterns = list(item["patterns"])
            dataset = SyntheticDataset(
                SyntheticConfig(
                    pattern=SyntheticPattern.TEXT_PREFIX,
                    text_patterns=patterns,
                    pattern_counts=dict(item["counts"]),
                    seed=42,
                )
            )
            n_tokens_target = sum(
                len(p.encode("utf-8")) * int(item["repeats"]) for p in patterns
            )
            if n_tokens_target >= 100_000:
                run_cfg = replace(self.run_config, max_steps=10, min_steps=4)
            else:
                run_cfg = replace(self.run_config, max_steps=14, min_steps=4)

            if self.dashboard:
                self.start_dashboard(
                    grid_size=run_cfg.grid_size,
                    run_name=f"{self.experiment_name}_{item['name']}",
                )
                state, run_meta = run_stream_on_manifold(
                    dataset.generate(),
                    config=run_cfg,
                    on_step=self.dashboard_update,
                )
                self.close_dashboard()
            else:
                state, run_meta = run_stream_on_manifold(
                    dataset.generate(), config=run_cfg
                )
            token_ids = state.get("token_ids")
            n_tokens_observed = (
                int(token_ids.numel())
                if token_ids is not None
                else int(run_meta["n_particles"])
            )

            result = self.inference.observe(
                state,
                scenario=str(item["name"]),
                run_name=str(item["name"]),
                n_tokens=n_tokens_observed,
                n_patterns=int(len(patterns)),
                families=int(item["families"]),
                branches_per_family=int(item["branches_per_family"]),
                repeats_per_pattern=int(item["repeats"]),
                expected_unique_edges=int(item["expected_unique_edges"]),
                expected_branch_nodes=int(item["expected_branch_nodes"]),
                expected_max_out_degree=int(item["expected_max_out_degree"]),
                run_backend=str(run_meta["run_backend"]),
                run_steps=int(run_meta["run_steps"]),
                run_termination=str(run_meta["run_termination"]),
                init_ms=float(run_meta["init_ms"]),
                simulate_ms=float(run_meta["simulate_ms"]),
            )
            expected_edges = int(item["expected_unique_edges"])
            expected_out_deg = int(item["expected_max_out_degree"])

            observed_edges = int(result.get("transition_unique_edges", 0) or 0)
            observed_out_deg = int(result.get("sql_max_out_degree", 0) or 0)
            edge_cov = (
                float(observed_edges / expected_edges) if expected_edges > 0 else 0.0
            )
            out_cov = (
                float(observed_out_deg / expected_out_deg)
                if expected_out_deg > 0
                else 0.0
            )

            result["branch_edge_coverage"] = edge_cov
            result["out_degree_coverage"] = out_cov
            self.inference.results[-1]["branch_edge_coverage"] = edge_cov
            self.inference.results[-1]["out_degree_coverage"] = out_cov

        return self.project()

    def observe(self, state: dict) -> dict:
        return self.inference.observe(state)

    def project(self) -> dict:
        return self.projector.project(self.inference)
