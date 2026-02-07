"""Collision experiment for paper-ready artifacts.

Goal: run a medium-scale (not toy, not full) controlled collision workload and
write outputs as tables + figures under `paper/` for direct inclusion in the
paper.
"""

from __future__ import annotations

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

        # Medium-scale scenarios (run through the actual Manifold wrapper).
        # 1) Controlled synthetic collision workload
        self.scenarios: list[tuple[str, DatasetProtocol]] = [
            (
                "medium_collision",
                SyntheticDataset(
                    SyntheticConfig(
                        pattern=SyntheticPattern.COLLISION,
                        num_units=128,
                        unit_length=512,
                        collision_rate=0.5,
                        seed=42,
                    )
                ),
            ),
            (
                "medium_repeated_ab",
                SyntheticDataset(
                    SyntheticConfig(
                        pattern=SyntheticPattern.REPEATED,
                        num_units=128,
                        unit_length=512,
                        repeat_sequence=b"AB",
                        seed=42,
                    )
                ),
            ),
            # 2) Real text stream at medium scale
            (
                "medium_wikitext",
                HuggingFaceDataset(
                    HuggingFaceConfig(
                        name="wikitext",
                        subset="wikitext-2-raw-v1",
                        split="train",
                        field="text",
                        streaming=True,
                        max_samples=512,
                    )
                ),
            ),
        ]
        self.inference = InferenceObserver(
            [
                TokenDistributionMetrics().observe,
                SpatialClustering().observe,
                WaveFieldMetrics().observe,
                MapPathMetrics(
                    key=KeySpec(kind="spatial_morton_byte"), topk=20
                ).observe,
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
                    caption="Collision experiment summary metrics",
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

    def run(self):
        for scenario_name, dataset in self.scenarios:
            tokenizer = UniversalTokenizer(
                datasets=[dataset],
                config=UniversalTokenizerConfig(
                    max_tokens=65536,
                    batch_tokens=65536,
                    seed=42,
                ),
            )
            instrumentation = []
            if self.dashboard:
                self.start_dashboard(
                    grid_size=self.grid_size,
                    run_name=f"{self.experiment_name}_{scenario_name}",
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
                run_name=f"{self.experiment_name}_{scenario_name}",
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
        return self.projector.project(self.inference)
