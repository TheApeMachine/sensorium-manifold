"""Scaled rule-shift experiment for reviewer-grade adaptation evidence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


from sensorium.experiments.base import Experiment
from sensorium.experiments.manifold_runner import (
    ManifoldRunConfig,
    run_stream_on_manifold,
)

# 1. DATASETS
from sensorium.dataset import (
    RuleShiftConfig,
    RuleShiftDataset,
)
from sensorium.dataset.rule_shift import RuleShiftPhase

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    RuleShiftPredictor,
    ParticleCount,
    ModeCount,
)

# 3. MANIFOLD
# 4. PROJECTORS
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.rule_shift import (
    RuleShiftTableProjector,
    RuleShiftFigureProjector,
    RuleShiftFigureConfig,
)


class KernelRuleShift(Experiment):
    """Rule-shift adaptation matrix with multiple conditions and seed repeats."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)

        self.vocab_size = 4096
        self.prime = 31
        self.context_length = 8
        self.eval_every = 5
        self.grid_size = (64, 64, 64)
        self.dt = 0.01
        self.seeds = tuple(self.experiment_seeds(default=(7, 19, 43)))
        self.run_config = ManifoldRunConfig(
            grid_size=self.grid_size,
            max_steps=16,
            min_steps=4,
            allow_analysis_fallback=False,
        )
        self.scenarios = self._build_scenarios()
        self._run_rows: list[dict[str, Any]] = []
        self._dashboard_tags: set[str] = set()
        self._provenance_path: str = ""

        self.inference = InferenceObserver(
            [ParticleCount().observe, ModeCount().observe]
        )

        self.predictor = RuleShiftPredictor(
            vocab_size=self.vocab_size,
            prime=self.prime,
            context_length=self.context_length,
        )

        self.projector = PipelineProjector(
            ConsoleProjector(),
            RuleShiftTableProjector(output_dir=Path("paper/tables")),
            RuleShiftFigureProjector(
                config=RuleShiftFigureConfig(name="rule_shift"),
                output_dir=Path("paper/figures"),
            ),
        )

    def _build_scenarios(self) -> tuple[dict, ...]:
        return (
            {
                "scenario": "toy_reverse_control",
                "domain": "synthetic-control",
                "segment_size": 24,
                "phases": (
                    RuleShiftPhase("forward", "The cat sat on the mat.", 60),
                    RuleShiftPhase("reverse", "mat the on sat cat The.", 60),
                ),
            },
            {
                "scenario": "style_swap_natural",
                "domain": "natural-language",
                "segment_size": 48,
                "phases": (
                    RuleShiftPhase(
                        "plain", "A calm river curves past the old stone bridge.", 70
                    ),
                    RuleShiftPhase(
                        "technical",
                        "Bridge stress gradients spike when resonance locks to wind.",
                        70,
                    ),
                ),
            },
            {
                "scenario": "three_phase_curriculum",
                "domain": "continual-shift",
                "segment_size": 56,
                "phases": (
                    RuleShiftPhase(
                        "news",
                        "Markets open mixed as energy and transit shares diverge.",
                        45,
                    ),
                    RuleShiftPhase(
                        "code",
                        "def route(packet): return table.get(packet.dst, default_hop)",
                        45,
                    ),
                    RuleShiftPhase(
                        "news_return",
                        "Markets open mixed as energy and transit shares diverge.",
                        45,
                    ),
                ),
            },
            {
                "scenario": "real_world_wikitext_focus",
                "domain": "real-world-text",
                "segment_size": 64,
                "phases": (
                    RuleShiftPhase(
                        "wiki_fact",
                        "The Apollo program landed humans on the Moon in 1969 and returned samples.",
                        90,
                    ),
                    RuleShiftPhase(
                        "wiki_technical",
                        "Saturn V guidance solved trajectory updates by fusing radar and inertial frames.",
                        90,
                    ),
                ),
            },
        )

    def run(self):
        """Run multi-scenario rule-shift matrix and project artifacts."""
        print(
            f"[rule_shift] Starting scaled matrix: {len(self.scenarios)} scenarios x {len(self.seeds)} seeds"
        )

        self._run_rows = []
        self._dashboard_tags = set()
        self._provenance_path = ""
        run_count = 0
        wall_times: list[float] = []
        n_particles_acc: list[int] = []
        n_modes_acc: list[int] = []

        for scenario_idx, scenario in enumerate(self.scenarios):
            cfg = RuleShiftConfig(
                phases=scenario["phases"],
                segment_size=int(scenario["segment_size"]),
            )
            for seed_idx, seed in enumerate(self.seeds):
                dataset = RuleShiftDataset(cfg)
                torch_mod = __import__("torch")
                torch_mod.manual_seed(int(seed))
                run_count += 1
                print(
                    f"[rule_shift] ({run_count}/{len(self.scenarios) * len(self.seeds)}) "
                    f"{scenario['scenario']} seed={seed} reps={dataset.total_reps}"
                )

                start_time = time.time()
                scenario_tag = str(scenario["scenario"])
                run_name = f"{self.experiment_name}_{scenario_tag}_s{int(seed)}"
                record_dashboard = (
                    bool(self.dashboard) and scenario_tag not in self._dashboard_tags
                )
                if record_dashboard:
                    self.start_dashboard(grid_size=self.grid_size, run_name=run_name)
                try:
                    state, meta = run_stream_on_manifold(
                        dataset.generate(),
                        config=self.run_config,
                        on_step=self.dashboard_update if record_dashboard else None,
                    )
                finally:
                    if record_dashboard:
                        self.close_dashboard()
                        self._dashboard_tags.add(scenario_tag)
                wall_time_ms = (time.time() - start_time) * 1000.0
                wall_times.append(float(wall_time_ms))
                self._run_rows.append(
                    {
                        "scenario": str(scenario["scenario"]),
                        "domain": str(scenario["domain"]),
                        "seed": int(seed),
                        "run_name": run_name,
                        "run_backend": str(meta.get("run_backend", "unknown")),
                        "run_steps": int(meta.get("run_steps", 0)),
                        "run_termination": str(meta.get("run_termination", "unknown")),
                        "init_ms": float(meta.get("init_ms", 0.0)),
                        "simulate_ms": float(meta.get("simulate_ms", 0.0)),
                        "wall_time_ms": float(wall_time_ms),
                        "n_phases": int(len(dataset.phases)),
                        "total_reps": int(dataset.total_reps),
                        "total_bytes": int(len(dataset.train_bytes)),
                    }
                )

                token_ids = state.get("token_ids")
                if token_ids is None:
                    continue

                n_particles = int(len(token_ids.cpu().numpy()))
                n_particles_acc.append(n_particles)
                n_modes = int(meta.get("run_steps", 0))
                n_modes_acc.append(n_modes)

                prediction = self.predictor.observe(
                    {
                        "token_ids": token_ids,
                        "sequence_indices": state.get("sequence_indices"),
                        "energies": state.get(
                            "energies", torch_mod.ones(len(token_ids))
                        ),
                        "phase_schedule": dataset.phase_schedule,
                        "forward_phrase": dataset.forward_phrase,
                        "reverse_phrase": dataset.reverse_phrase,
                        "forward_reps": dataset.forward_reps,
                        "reverse_reps": dataset.reverse_reps,
                        "eval_every": self.eval_every,
                        "segment_size": dataset.segment_size,
                        "phase_switch_byte": dataset.phase_switch_byte,
                    }
                )

                self.inference.observe(
                    state,
                    manifold=None,
                    run_name=run_name,
                    run_backend=str(meta.get("run_backend", "unknown")),
                    run_steps=int(meta.get("run_steps", 0)),
                    run_termination=str(meta.get("run_termination", "unknown")),
                    init_ms=float(meta.get("init_ms", 0.0)),
                    simulate_ms=float(meta.get("simulate_ms", 0.0)),
                    scenario=str(scenario["scenario"]),
                    domain=str(scenario["domain"]),
                    seed=int(seed),
                    n_phases=int(len(dataset.phases)),
                    total_reps=int(dataset.total_reps),
                    total_bytes=int(len(dataset.train_bytes)),
                    segment_size=int(dataset.segment_size),
                    context_length=int(self.context_length),
                    eval_every=int(self.eval_every),
                    vocab_size=int(self.vocab_size),
                    hash_prime=int(self.prime),
                    wall_time_ms=float(wall_time_ms),
                    phase_schedule=dataset.phase_schedule,
                    **prediction,
                )

        self.assert_allowed_backends(self._run_rows, allowed=("mps",))
        self._provenance_path = str(
            self.write_provenance_jsonl(
                self._run_rows, stem=f"{self.experiment_name}_raw"
            )
        )
        for row in self.inference.results:
            row["provenance_jsonl"] = self._provenance_path
        self.project()

        mean_particles = int(sum(n_particles_acc) / max(1, len(n_particles_acc)))
        mean_modes = int(sum(n_modes_acc) / max(1, len(n_modes_acc)))
        mean_wall_ms = float(sum(wall_times) / max(1, len(wall_times)))
        print(
            "[rule_shift] Completed scaled matrix: "
            f"runs={run_count}, mean_particles={mean_particles}, "
            f"mean_modes={mean_modes}, mean_wall_ms={mean_wall_ms:.1f}"
        )

    def observe(self, state: dict) -> dict:
        """Observer interface for compatibility."""
        return {}

    def project(self) -> dict:
        """Project observation to artifacts."""
        outputs = self.projector.project(self.inference)
        return {"projectors": outputs, "provenance_jsonl": self._provenance_path}
