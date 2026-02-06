"""Ablations for keying strategy and sequence recoverability."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensorium.dataset import SyntheticConfig, SyntheticDataset, SyntheticPattern
from sensorium.experiments.base import Experiment
from sensorium.experiments.state_builder import StateBuildConfig, build_observation_state
from sensorium.observers.inference import InferenceObserver
from sensorium.projectors import ConsoleProjector, LaTeXTableProjector, PipelineProjector, TableConfig


class KeyAblationObserver:
    """Compare information retained by different key definitions."""

    def observe(self, state: dict | None = None, **kwargs) -> dict:
        if not isinstance(state, dict):
            return {}

        token_ids = state.get("token_ids")
        byte_values = state.get("byte_values")
        seq_idx = state.get("sequence_indices")
        sample_idx = state.get("sample_indices")
        if token_ids is None or byte_values is None or seq_idx is None or sample_idx is None:
            return {}

        tok = token_ids.detach().to("cpu").numpy()
        byt = byte_values.detach().to("cpu").numpy()
        seq = seq_idx.detach().to("cpu").numpy()
        sam = sample_idx.detach().to("cpu").numpy()

        n = int(tok.size)
        if n <= 1:
            return {
                "n_tokens": n,
                "unique_sequence_byte": int(np.unique(tok).size),
                "unique_byte_only": int(np.unique(byt).size),
                "edge_recall_byte_only": 0.0,
                "edge_recall_sequence_byte": 1.0,
            }

        order = np.lexsort((seq, sam))
        sam_o = sam[order]
        seq_o = seq[order]
        tok_o = tok[order]
        byt_o = byt[order]
        ok = (sam_o[1:] == sam_o[:-1]) & (seq_o[1:] == seq_o[:-1] + 1)
        src_tok = tok_o[:-1][ok]
        dst_tok = tok_o[1:][ok]
        src_byt = byt_o[:-1][ok]
        dst_byt = byt_o[1:][ok]

        edge_tok = (src_tok.astype(np.uint64) << np.uint64(32)) | (dst_tok.astype(np.uint64) & np.uint64(0xFFFFFFFF))
        edge_byt = (src_byt.astype(np.uint64) << np.uint64(8)) | (dst_byt.astype(np.uint64) & np.uint64(0xFF))

        uniq_tok = int(np.unique(edge_tok).size)
        uniq_byt = int(np.unique(edge_byt).size)
        edge_recall = float(uniq_byt / max(1, uniq_tok))

        return {
            "n_tokens": n,
            "unique_sequence_byte": int(np.unique(tok).size),
            "unique_byte_only": int(np.unique(byt).size),
            "compression_sequence_byte": float(np.unique(tok).size / max(1, n)),
            "compression_byte_only": float(np.unique(byt).size / max(1, n)),
            "edge_recall_byte_only": edge_recall,
            "edge_recall_sequence_byte": 1.0,
        }


class AblationsExperiment(Experiment):
    """Run key-space ablations used to rebut toy-scale/path-loss objections."""

    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=[
                "scenario",
                "n_tokens",
                "unique_sequence_byte",
                "unique_byte_only",
                "compression_sequence_byte",
                "compression_byte_only",
                "edge_recall_sequence_byte",
                "edge_recall_byte_only",
            ],
        )

        self.state_config = StateBuildConfig(grid_size=(64, 64, 64), mode_bins=256)
        self.inference = InferenceObserver([KeyAblationObserver()])
        self.projector = PipelineProjector(
            ConsoleProjector(fields=self.reportable, format="table"),
            LaTeXTableProjector(
                TableConfig(
                    name="ablation_summary",
                    columns=self.reportable,
                    caption="Keying ablations: sequence+byte vs byte-only",
                    label="tab:ablation",
                    precision=4,
                ),
                output_dir=Path("paper/tables"),
            ),
        )

    def run(self):
        scenarios = [
            ("small_prefix", 64, 128),
            ("medium_prefix", 128, 512),
            ("large_prefix", 256, 1024),
        ]
        for name, units, length in scenarios:
            dataset = SyntheticDataset(
                SyntheticConfig(
                    pattern=SyntheticPattern.COLLISION,
                    num_units=int(units),
                    unit_length=int(length),
                    collision_rate=0.5,
                    seed=42,
                )
            )
            state = build_observation_state(dataset.generate(), config=self.state_config)
            self.inference.observe(state, scenario=name)

        return self.project()

    def observe(self, state: dict):
        return self.inference.observe(state)

    def project(self):
        return self.projector.project(self.inference)
