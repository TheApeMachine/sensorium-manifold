"""Wave-trie / holographic associative memory (easy proof).

This experiment does NOT change physics. It "shows the cards" by extracting a
lossless path representation (sequence transitions) and a compressed map view
(folded mass over content+position keys), then emitting paper-ready artifacts.

Key idea:
- Map key is not a hash: (sequence_position, byte_value)
  This is the simplest lossless ID for the "what" + "where (claimed)" concept.
- Path transitions are measured over these keys within each sample.
"""

from __future__ import annotations

from sensorium.experiments.base import Experiment
from sensorium.manifold import Manifold
from sensorium.tokenizer.universal import UniversalTokenizer
from sensorium.observers.sql import SQLObserver
from sensorium.observers.inference import InferenceObserver
from sensorium.dataset import HuggingFaceDataset, HuggingFaceConfig


class WaveTrieExperiment(Experiment):
    """Measure position+byte folding + transitions (easy proof)."""

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(
            experiment_name, profile, dashboard=dashboard,
            reportable=[
                "scenario",
                "n_tokens",
                "n_samples",
                "unique_keys",
                "key_collision_rate",
                "transition_edges",
                "transition_unique_edges",
                "transition_top1_prob",
                "transition_entropy",
                "fold_top1",
                "fold_pr",
                "fold_entropy",
                "psi_delta_rel",
            ]
        )

        self.manifold = Manifold(
            tokenizer=UniversalTokenizer(datasets=[HuggingFaceDataset(HuggingFaceConfig(
                name="nyu-mll/glue",
                subset="mrpc",
                split="train",
                field="text",
                streaming=True,
            ))])
        )

    def observe(self, _state: dict) -> dict:
        return InferenceObserver(
            SQLObserver("""
                SELECT 
                    CAST(particles AS TEXT), 
                    load_count 
                FROM simulation;
                JOIN simulation.thermodynamic ON simulation.id = simulation.thermodynamic.id;
                JOIN simulation.coherence ON simulation.id = simulation.coherence.id;
                WHERE simulation.coherence.state = 'CRYSTALLIZED'
                ORDER BY simulation.coherence.amplitude DESC 
                LIMIT 5;
            """)
        )

    def run(self):
        self.project(self.observe(self.manifold.run()))

    def project(self) -> dict:
        return self.projector.project(self.inference)

