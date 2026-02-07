"""Transparent scaling experiment matrix for the Sensorium manifold."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sensorium.dataset import (
    GeneralizationType,
    ScalingDataset,
    ScalingDatasetConfig,
    ScalingTestType,
)
from sensorium.experiments.base import Experiment
from sensorium.experiments.manifold_runner import (
    ManifoldRunConfig,
    run_stream_on_manifold,
)
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import ModeCount, ParticleCount
from sensorium.projectors import ConsoleProjector, PipelineProjector
from sensorium.projectors.scaling import (
    ScalingComputeFigureProjector,
    ScalingDynamicsFigureProjector,
    ScalingTableProjector,
)


class KernelScaling(Experiment):
    """Reviewer-grade scaling experiment with seed uncertainty reporting."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)

        self.seeds = (7, 19, 43)
        self.particle_counts = (1000, 2000, 4000, 8000, 16000)
        self.grid_sizes = ((16, 16, 16), (32, 32, 32), (64, 64, 64))
        self.pattern_counts = (1, 2, 4, 8, 16, 32)
        self.sequence_lengths = (1000, 4000, 16000, 64000)

        self.inference = InferenceObserver(
            [ParticleCount().observe, ModeCount().observe]
        )
        self.projector = PipelineProjector(
            ConsoleProjector(),
            ScalingTableProjector(output_dir=Path("paper/tables")),
            ScalingDynamicsFigureProjector(output_dir=Path("paper/figures")),
            ScalingComputeFigureProjector(output_dir=Path("paper/figures")),
        )

    def run(self):
        print("[scaling] Starting transparent scaling matrix...")
        results: dict[str, Any] = {
            "conditions": {
                "seeds": list(self.seeds),
                "particle_counts": list(self.particle_counts),
                "grid_sizes": [list(g) for g in self.grid_sizes],
                "pattern_counts": list(self.pattern_counts),
                "sequence_lengths": list(self.sequence_lengths),
            },
            "population": self._run_population_dynamics(),
            "interference": self._run_interference_test(),
            "compute": self._run_compute_scaling(),
            "latency": self._run_latency_test(),
            "generalization": self._run_generalization_test(),
        }
        self.inference.observe({}, **results)
        self.project()
        print("[scaling] Done.")

    def _run_once(
        self,
        dataset: ScalingDataset,
        *,
        grid_size: tuple[int, int, int],
        max_steps: int,
    ):
        history: dict[str, list[float]] = {
            "step": [],
            "n_modes": [],
            "n_volatile": [],
            "n_stable": [],
            "n_crystallized": [],
            "n_births": [],
            "n_deaths": [],
            "conflict_score": [],
        }
        prev_modes = 0

        def on_step(state: dict):
            nonlocal prev_modes
            amps = state.get("amplitudes")
            mstate = state.get("mode_state")
            conflict = state.get("conflict")
            step = int(state.get("step", 0))

            n_modes = 0
            n_volatile = 0
            n_stable = 0
            n_crystallized = 0
            conflict_score = 0.0
            if amps is not None:
                active = amps > 1e-6
                n_modes = int(active.sum().item())
                if mstate is not None and n_modes > 0:
                    states = mstate[:n_modes]
                    n_volatile = int((states == 0).sum().item())
                    n_stable = int((states == 1).sum().item())
                    n_crystallized = int((states == 2).sum().item())
                if conflict is not None and n_modes > 0:
                    conflict_score = float(conflict[:n_modes].mean().item())
            n_births = max(0, n_modes - prev_modes)
            n_deaths = max(0, prev_modes - n_modes)
            prev_modes = n_modes

            history["step"].append(step)
            history["n_modes"].append(float(n_modes))
            history["n_volatile"].append(float(n_volatile))
            history["n_stable"].append(float(n_stable))
            history["n_crystallized"].append(float(n_crystallized))
            history["n_births"].append(float(n_births))
            history["n_deaths"].append(float(n_deaths))
            history["conflict_score"].append(float(conflict_score))

        state, meta = run_stream_on_manifold(
            dataset.generate(),
            config=ManifoldRunConfig(
                grid_size=grid_size,
                max_steps=max_steps,
                min_steps=min(4, max_steps),
            ),
            on_step=on_step,
        )
        return state, meta, history

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    @staticmethod
    def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        lx = np.log(np.maximum(x, 1e-12))
        ly = np.log(np.maximum(y, 1e-12))
        alpha, log_a = np.polyfit(lx, ly, 1)
        return float(alpha), float(math.exp(log_a))

    def _run_population_dynamics(self) -> dict[str, Any]:
        histories: list[dict[str, list[float]]] = []
        finals: list[dict[str, float]] = []
        for seed in self.seeds:
            ds = ScalingDataset(
                ScalingDatasetConfig(
                    test_type=ScalingTestType.POPULATION,
                    n_bytes=4000,
                    seed=int(seed),
                )
            )
            _, meta, history = self._run_once(ds, grid_size=(32, 32, 32), max_steps=300)
            histories.append(history)
            births = float(np.sum(history["n_births"]))
            deaths = float(np.sum(history["n_deaths"]))
            finals.append(
                {
                    "seed": float(seed),
                    "steps": float(meta.get("run_steps", 0)),
                    "n_modes_final": float(
                        history["n_modes"][-1] if history["n_modes"] else 0.0
                    ),
                    "n_crystallized_final": float(
                        history["n_crystallized"][-1]
                        if history["n_crystallized"]
                        else 0.0
                    ),
                    "pruning_rate": float(deaths / (births + 1.0)),
                }
            )

        min_len = min((len(h["step"]) for h in histories), default=0)
        mean_history: dict[str, list[float]] = {}
        std_history: dict[str, list[float]] = {}
        keys = [
            "step",
            "n_modes",
            "n_volatile",
            "n_stable",
            "n_crystallized",
            "n_births",
            "n_deaths",
            "conflict_score",
        ]
        for key in keys:
            if key == "step":
                mean_history[key] = (
                    [float(v) for v in histories[0][key][:min_len]] if histories else []
                )
                std_history[key] = [0.0] * min_len
            else:
                mat = np.asarray(
                    [h[key][:min_len] for h in histories], dtype=np.float64
                )
                mean_history[key] = (
                    [float(v) for v in np.mean(mat, axis=0)] if mat.size else []
                )
                std_history[key] = (
                    [float(v) for v in np.std(mat, axis=0)] if mat.size else []
                )

        return {
            "history_mean": mean_history,
            "history_std": std_history,
            "seed_rows": finals,
            "n_modes_final_mean": float(np.mean([r["n_modes_final"] for r in finals]))
            if finals
            else 0.0,
            "n_crystallized_final_mean": float(
                np.mean([r["n_crystallized_final"] for r in finals])
            )
            if finals
            else 0.0,
            "pruning_rate_mean": float(np.mean([r["pruning_rate"] for r in finals]))
            if finals
            else 0.0,
        }

    def _run_interference_test(self) -> dict[str, Any]:
        rows: list[dict[str, float]] = []
        for n_patterns in self.pattern_counts:
            per_seed: list[dict[str, float]] = []
            for seed in self.seeds:
                ds = ScalingDataset(
                    ScalingDatasetConfig(
                        test_type=ScalingTestType.INTERFERENCE,
                        n_bytes=4000,
                        n_patterns=int(n_patterns),
                        seed=int(seed),
                    )
                )
                _, _, hist = self._run_once(ds, grid_size=(32, 32, 32), max_steps=220)
                cryst = float(
                    hist["n_crystallized"][-1] if hist["n_crystallized"] else 0.0
                )
                conf = (
                    float(np.mean(hist["conflict_score"]))
                    if hist["conflict_score"]
                    else 0.0
                )
                per_seed.append(
                    {
                        "n_crystallized": cryst,
                        "conflict": conf,
                        "efficiency": float(cryst / max(1, n_patterns)),
                    }
                )
            rows.append(
                {
                    "n_patterns": float(n_patterns),
                    "n_crystallized_mean": float(
                        np.mean([s["n_crystallized"] for s in per_seed])
                    ),
                    "n_crystallized_std": float(
                        np.std([s["n_crystallized"] for s in per_seed])
                    ),
                    "conflict_mean": float(np.mean([s["conflict"] for s in per_seed])),
                    "conflict_std": float(np.std([s["conflict"] for s in per_seed])),
                    "efficiency_mean": float(
                        np.mean([s["efficiency"] for s in per_seed])
                    ),
                    "efficiency_std": float(
                        np.std([s["efficiency"] for s in per_seed])
                    ),
                }
            )
        return {"rows": rows}

    def _run_compute_scaling(self) -> dict[str, Any]:
        particle_rows: list[dict[str, float]] = []
        for n_particles in self.particle_counts:
            vals: list[float] = []
            for seed in self.seeds:
                ds = ScalingDataset(
                    ScalingDatasetConfig(
                        test_type=ScalingTestType.COMPUTE,
                        n_bytes=int(n_particles),
                        seed=int(seed),
                    )
                )
                _, meta, _ = self._run_once(ds, grid_size=(32, 32, 32), max_steps=16)
                steps = max(1.0, float(meta.get("run_steps", 0)))
                vals.append(float(meta.get("simulate_ms", 0.0)) / steps)
            mu, sd = self._mean_std(vals)
            particle_rows.append(
                {
                    "n_particles": float(n_particles),
                    "ms_per_step_mean": mu,
                    "ms_per_step_std": sd,
                }
            )

        grid_rows: list[dict[str, Any]] = []
        for grid_size in self.grid_sizes:
            vals: list[float] = []
            for seed in self.seeds:
                ds = ScalingDataset(
                    ScalingDatasetConfig(
                        test_type=ScalingTestType.COMPUTE,
                        n_bytes=4000,
                        seed=int(seed),
                    )
                )
                _, meta, _ = self._run_once(ds, grid_size=grid_size, max_steps=16)
                steps = max(1.0, float(meta.get("run_steps", 0)))
                vals.append(float(meta.get("simulate_ms", 0.0)) / steps)
            mu, sd = self._mean_std(vals)
            grid_rows.append(
                {
                    "grid_size": list(grid_size),
                    "grid_cells": float(grid_size[0] * grid_size[1] * grid_size[2]),
                    "ms_per_step_mean": mu,
                    "ms_per_step_std": sd,
                }
            )

        x = np.asarray([r["n_particles"] for r in particle_rows], dtype=np.float64)
        y = np.asarray([r["ms_per_step_mean"] for r in particle_rows], dtype=np.float64)
        alpha, coeff = self._fit_power_law(x, y)
        return {
            "by_particles": particle_rows,
            "by_grid": grid_rows,
            "particle_fit_alpha": float(alpha),
            "particle_fit_coeff": float(coeff),
        }

    def _run_latency_test(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        per_len_samples: dict[int, list[float]] = {}
        for seq_len in self.sequence_lengths:
            vals: list[float] = []
            for seed in self.seeds:
                ds = ScalingDataset(
                    ScalingDatasetConfig(
                        test_type=ScalingTestType.LATENCY,
                        n_bytes=int(seq_len),
                        seed=int(seed),
                    )
                )
                _, meta, _ = self._run_once(ds, grid_size=(32, 32, 32), max_steps=16)
                steps = max(1.0, float(meta.get("run_steps", 0)))
                vals.append(float(meta.get("simulate_ms", 0.0)) / steps)
            per_len_samples[int(seq_len)] = vals
            mu, sd = self._mean_std(vals)
            rows.append(
                {
                    "seq_len": float(seq_len),
                    "ms_per_step_mean": mu,
                    "ms_per_step_std": sd,
                    "cv": float(sd / mu) if mu > 0.0 else 0.0,
                }
            )

        x = np.asarray([r["seq_len"] for r in rows], dtype=np.float64)
        y = np.asarray([r["ms_per_step_mean"] for r in rows], dtype=np.float64)
        alpha, coeff = self._fit_power_law(x, y)

        boot: list[float] = []
        rng = np.random.default_rng(123)
        for _ in range(400):
            sample_y = []
            for seq_len in self.sequence_lengths:
                vals = per_len_samples[int(seq_len)]
                sample_y.append(float(vals[int(rng.integers(0, len(vals)))]))
            b_alpha, _ = self._fit_power_law(x, np.asarray(sample_y, dtype=np.float64))
            boot.append(float(b_alpha))
        ci_low, ci_high = np.percentile(np.asarray(boot, dtype=np.float64), [2.5, 97.5])

        return {
            "rows": rows,
            "fit_alpha": float(alpha),
            "fit_coeff": float(coeff),
            "fit_alpha_ci_low": float(ci_low),
            "fit_alpha_ci_high": float(ci_high),
            "cv_mean": float(np.mean([r["cv"] for r in rows])) if rows else 0.0,
        }

    def _run_generalization_test(self) -> dict[str, Any]:
        cases = (
            ("repetitive", GeneralizationType.REPETITIVE),
            ("semi_random", GeneralizationType.SEMI_RANDOM),
            ("natural_like", GeneralizationType.NATURAL_LIKE),
            ("pure_random", GeneralizationType.PURE_RANDOM),
        )
        rows: list[dict[str, Any]] = []
        for name, gtype in cases:
            structures: list[float] = []
            entropies: list[float] = []
            collisions: list[float] = []
            for seed in self.seeds:
                ds = ScalingDataset(
                    ScalingDatasetConfig(
                        test_type=ScalingTestType.GENERALIZATION,
                        generalization_type=gtype,
                        n_bytes=4000,
                        seed=int(seed),
                    )
                )
                state, _, hist = self._run_once(
                    ds, grid_size=(32, 32, 32), max_steps=180
                )
                cryst = float(
                    hist["n_crystallized"][-1] if hist["n_crystallized"] else 0.0
                )
                structures.append(float(cryst / 64.0))
                token_ids = state.get("token_ids")
                if token_ids is None:
                    entropies.append(1.0)
                    collisions.append(1.0)
                    continue
                tid = token_ids.detach().cpu().numpy()
                uniq, cnt = np.unique(tid, return_counts=True)
                p = cnt / max(1, cnt.sum())
                entropy = float(-np.sum(p * np.log2(p + 1e-10)))
                max_entropy = float(np.log2(max(1, len(uniq))))
                entropies.append(
                    float(entropy / max_entropy) if max_entropy > 0 else 0.0
                )
                collisions.append(float(len(tid) / max(1, len(uniq))))

            s_mu, s_sd = self._mean_std(structures)
            e_mu, e_sd = self._mean_std(entropies)
            c_mu, c_sd = self._mean_std(collisions)
            rows.append(
                {
                    "name": name,
                    "structure_ratio_mean": s_mu,
                    "structure_ratio_std": s_sd,
                    "normalized_entropy_mean": e_mu,
                    "normalized_entropy_std": e_sd,
                    "collision_ratio_mean": c_mu,
                    "collision_ratio_std": c_sd,
                }
            )
        return {"rows": rows}

    def observe(self, state: dict):
        pass

    def project(self) -> dict:
        return self.projector.project(self.inference)
