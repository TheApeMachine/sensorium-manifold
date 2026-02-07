"""Collision experiment for paper-ready artifacts.

Goal: run a medium-scale (not toy, not full) controlled collision workload and
write outputs as tables + figures under `paper/` for direct inclusion in the
paper.
"""

from __future__ import annotations

from dataclasses import replace
from sensorium.experiments.base import (
    Experiment,
)
import time

from sensorium.dataset import (
    HuggingFaceConfig,
    HuggingFaceDataset,
    SyntheticConfig,
    SyntheticDataset,
    SyntheticPattern,
)
from sensorium.dataset.base import DatasetProtocol
from sensorium.manifold import Manifold
from sensorium.tokenizer.universal import UniversalTokenizer, UniversalTokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    TokenDistributionMetrics,
    SpatialClustering,
    WaveFieldMetrics,
    MapPathMetrics,
    KeySpec,
    CollisionPaperArtifacts,
)
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
    TableConfig,
    LaTeXTableProjector,
    TopTransitionsProjector,
    CollisionFigureConfig,
    CollisionFigureProjector,
)

from sensorium.console import console


class CollisionExperiment(Experiment):
    """Demonstrate Thermodynamic Trie (Collision is Compression)

    To avoid further confusion, when we talk about "collision" in this experiment,
    we mean physical collision of particles, no hash collisions.
    In all fairness, we did indeed have a hash collision issue, however that was a spur
    of the moment idea that we were in no way dependent on. And so, we have since just
    moved to still using raw bytes, just with the sequence index weaved in.
    Problem solved. Collision is Compression, yet again (but kinda always was anyway).
    """

    def __init__(
        self, experiment_name: str, profile: bool = False, dashboard: bool = False
    ):
        reportable = [
            "scenario",
            "seed",
            "n_tokens",
            "n_samples",
            "n_unique_tokens",
            "compression_ratio",
            "collision_rate_observed",
            "entropy",
            "unique_keys",
            "key_collision_rate",
            "fold_top1",
            "fold_pr",
            "transition_top1_prob",
            "transition_unique_edges",
            "recall_top1",
            "recall_top3",
            "recall_mrr",
            "spatial_clustering",
            "mode_participation",
            "mode_entropy",
            "psi_delta_rel",
            "run_backend",
            "run_steps",
            "run_termination",
            "simulate_ms",
        ]
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=reportable,
        )

        self.grid_size: tuple[int, int, int] = (64, 64, 64)
        self.max_steps = 12
        self.min_steps = 4
        self.seeds = self.experiment_seeds()

        # Medium-scale scenarios (run through the actual Manifold wrapper).
        self.scenario_specs: list[tuple[str, object]] = [
            (
                "medium_collision",
                SyntheticConfig(
                    pattern=SyntheticPattern.COLLISION,
                    num_units=128,
                    unit_length=512,
                    collision_rate=0.5,
                    seed=42,
                ),
            ),
            (
                "medium_repeated_ab",
                SyntheticConfig(
                    pattern=SyntheticPattern.REPEATED,
                    num_units=128,
                    unit_length=512,
                    repeat_sequence=b"AB",
                    seed=42,
                ),
            ),
            (
                "medium_wikitext",
                HuggingFaceConfig(
                    name="wikitext",
                    subset="wikitext-2-raw-v1",
                    split="train",
                    field="text",
                    streaming=True,
                    max_samples=512,
                ),
            ),
        ]
        self.inference = InferenceObserver(
            [
                TokenDistributionMetrics().observe,
                SpatialClustering().observe,
                WaveFieldMetrics().observe,
                MapPathMetrics(key=KeySpec(kind="sequence_byte"), topk=20).observe,
                CollisionPaperArtifacts().observe,
            ]
        )

        # Collision experiment needs paper-ready tables + figures.
        self.projector = PipelineProjector(
            ConsoleProjector(fields=reportable, format="table"),
            TopTransitionsProjector(),
            LaTeXTableProjector(
                TableConfig(
                    name=f"{experiment_name}_summary",
                    columns=reportable,
                    caption="Collision experiment raw per-seed metrics (MPS only)",
                    label=f"tab:{experiment_name}",
                    precision=3,
                ),
                output_dir=self.artifact_path("tables"),
            ),
            CollisionFigureProjector(
                CollisionFigureConfig(
                    name_prefix=experiment_name, formats=("pdf",), dpi=220
                ),
                output_dir=self.artifact_path("figures"),
            ),
        )
        self.summary_table = LaTeXTableProjector(
            TableConfig(
                name=f"{experiment_name}_summary_ci",
                columns=[],
                caption="Collision experiment multi-seed summary with 95\\% CI (MPS only)",
                label=f"tab:{experiment_name}_ci",
                precision=4,
            ),
            output_dir=self.artifact_path("tables"),
        )

    def _build_dataset(self, spec: object, *, seed: int) -> DatasetProtocol:
        if isinstance(spec, SyntheticConfig):
            return SyntheticDataset(replace(spec, seed=int(seed)))
        if isinstance(spec, HuggingFaceConfig):
            return HuggingFaceDataset(spec)
        raise TypeError(f"Unsupported scenario spec type: {type(spec)!r}")

    def run(self):
        console.info("Running collision experiment")
        console.info(f"Seeds: {self.seeds}")

        for scenario_name, spec in self.scenario_specs:
            for seed in self.seeds:
                console.info(f"Running scenario: {scenario_name} (seed={seed})")
                dataset = self._build_dataset(spec, seed=seed)

                tokenizer = UniversalTokenizer(
                    datasets=[dataset],
                    config=UniversalTokenizerConfig(
                        max_tokens=65536,
                        batch_tokens=65536,
                        seed=int(seed),
                    ),
                )
                instrumentation = []
                if self.dashboard:
                    self.start_dashboard(
                        grid_size=self.grid_size,
                        run_name=f"{self.experiment_name}_{scenario_name}_s{seed}",
                    )
                    instrumentation = (
                        [self._dashboard_instance]
                        if self._dashboard_instance is not None
                        else []
                    )

                manifold = Manifold(
                    tokenizer=tokenizer,
                    grid_size=self.grid_size,
                    max_steps=self.max_steps,
                    instrumentation=instrumentation,
                )

                t0 = time.perf_counter()
                state = manifold.run()
                t1 = time.perf_counter()
                steps = int(state.get("step", 0))

                termination = "budget" if steps >= int(self.max_steps) else "quiet"
                if steps < int(self.min_steps):
                    termination = "quiet"  # only a labeling detail

                self.observe(
                    state,
                    scenario=str(scenario_name),
                    seed=int(seed),
                    run_name=f"{self.experiment_name}_{scenario_name}_s{seed}",
                    run_backend=str(manifold.device_name),
                    run_steps=steps,
                    run_termination=termination,
                    simulate_ms=(t1 - t0) * 1000.0,
                )

                if self.dashboard:
                    self.close_dashboard()

        return self.project()

    def observe(self, state: dict, **meta) -> dict:
        """Observe the manifold state using composed observers."""
        console.info("Observing manifold state")
        grid_dims = state.get("grid_size")
        if not (isinstance(grid_dims, (tuple, list)) and len(grid_dims) == 3):
            grid_dims = self.grid_size
        return self.inference.observe(
            state,
            hash_vocab_size=4096,
            grid_dims=tuple(int(x) for x in grid_dims),
            **meta,
        )

    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        console.info("Projecting accumulated observations to artifacts")
        rows = list(self.inference.results)
        self.assert_allowed_backends(rows, allowed=("mps",))
        provenance = self.write_provenance_jsonl(
            rows, stem=f"{self.experiment_name}_raw"
        )

        raw_outputs = self.projector.project(self.inference)

        metric_fields = self.infer_numeric_fields(
            rows,
            candidate_fields=self.reportable,
            exclude_fields=("seed",),
        )
        summary_rows = self.aggregate_rows_with_ci(
            rows,
            group_field="scenario",
            metric_fields=metric_fields,
            carry_fields=("run_backend",),
            seed_field="seed",
        )
        summary_columns = self.ci_summary_columns(
            metric_fields,
            prefix_fields=("scenario", "run_backend", "n_seeds"),
        )
        self.summary_table.config.columns = summary_columns
        summary_output = self.summary_table.project({"results": summary_rows})

        return {
            "raw": raw_outputs,
            "summary_ci": summary_output,
            "provenance_jsonl": str(provenance),
        }
