"""Collision experiment for paper-ready artifacts.

Goal: run a medium-scale (not toy, not full) controlled collision workload and
write outputs as tables + figures under `paper/` for direct inclusion in the
paper.
"""

from __future__ import annotations

from sensorium.experiments.base import (
    Experiment,
)

from sensorium.manifold import Manifold
from sensorium.dataset import (
    HuggingFaceConfig,
    HuggingFaceDataset,
)
from sensorium.tokenizer.universal import UniversalTokenizer
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.sql import SQLObserver
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
from sensorium.instrument.history import StateHistoryInstrument


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
        self, 
        experiment_name: str, 
        profile: bool = False, 
        dashboard: bool = False
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

        self.datasets = [
            HuggingFaceDataset(HuggingFaceConfig(
                name="wikitext",
                subset="wikitext-2-raw-v1",
                split="train",
                field="text",
                streaming=True,
                max_samples=1024,
            ))
        ]

        self.history = StateHistoryInstrument()

        self.manifold = Manifold(
            tokenizer=UniversalTokenizer(
                datasets=self.datasets
            ),
            instrumentation=[self.history]
        )
        
        self.inference = InferenceObserver(
            manifold=self.manifold,
            observers=[
                SQLObserver(
                    sql_query="""
                    SELECT 
                        simulation.particles, 
                        history.particles AS n_inputs
                    FROM 
                        simulation
                    WHERE
                        simulation.step = (
                            SELECT 
                                MAX(simulation.step) 
                            FROM 
                                simulation
                        )
                    """
                )
            ]
        )

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
                output_dir=f"{experiment_name}/tables",
            ),
            CollisionFigureProjector(
                CollisionFigureConfig(
                    name_prefix=experiment_name, formats=("pdf",), dpi=220
                ),
                output_dir=f"{experiment_name}/figures",
            ),
            LaTeXTableProjector(
                TableConfig(
                    name=f"{experiment_name}_summary_ci",
                    columns=[],
                    caption="Collision experiment multi-seed summary with 95\\% CI (MPS only)",
                    label=f"tab:{experiment_name}_ci",
                    precision=4,
                ),
                output_dir=f"{experiment_name}/tables",
            )
        )

    def run(self):
        console.info("Running collision experiment")

        return self.project(self.observe(self.manifold.run()))

    def observe(self, state: dict, **meta) -> dict:
        """Observe the manifold state using composed observers."""
        return self.inference.observe(
            state,
            manifold=self.manifold,
            **meta,
        )

    def project(self, observation: dict) -> dict:
        """Project accumulated observations to artifacts."""
        console.info("Projecting accumulated observations to artifacts")
        self.projector.project(observation)