"""Kernel-based rule-shift experiment.

This experiment uses the clean composable pattern:
- Datasets: RuleShiftDataset (forward then reverse phrase)
- Observers: RuleShiftPredictor, ParticleCount, ModeCount
- Projectors: RuleShiftTableProjector, RuleShiftFigureProjector

Produces:
- `paper/tables/rule_shift_summary.tex`
- `paper/figures/rule_shift.png`
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sensorium.experiments.base import Experiment

# 1. DATASETS
from sensorium.dataset import (
    RuleShiftConfig,
    RuleShiftDataset,
)

# 2. OBSERVERS
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    RuleShiftPredictor,
    ParticleCount,
    ModeCount,
)

# 3. MANIFOLD
from optimizer.manifold import Manifold, SimulationConfig, CoherenceSimulationConfig
from optimizer.tokenizer import TokenizerConfig

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
    """Rule-shift adaptation experiment using thermodynamic trie.
    
    Clean pattern:
    - datasets: RuleShiftDataset
    - manifold: Runs simulation
    - inference: InferenceObserver with RuleShiftPredictor
    - projector: RuleShiftTableProjector, RuleShiftFigureProjector
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Configuration
        self.vocab_size = 4096
        self.prime = 31
        self.context_length = 8
        self.eval_every = 5
        
        # 1. DATASET
        self.dataset = RuleShiftDataset(RuleShiftConfig(
            forward_reps=50,
            reverse_reps=50,
        ))
        
        # 2. INFERENCE OBSERVER
        self.inference = InferenceObserver(
            ParticleCount(),
            ModeCount(),
        )
        
        # 3. PREDICTOR (used separately for evaluation)
        self.predictor = RuleShiftPredictor(
            vocab_size=self.vocab_size,
            prime=self.prime,
            context_length=self.context_length,
        )
        
        # 4. PROJECTORS
        self.projector = PipelineProjector(
            ConsoleProjector(),
            RuleShiftTableProjector(output_dir=Path("paper/tables")),
            RuleShiftFigureProjector(
                config=RuleShiftFigureConfig(name="rule_shift"),
                output_dir=Path("paper/figures"),
            ),
        )

    def run(self):
        """Run the rule shift experiment."""
        print(f"[rule_shift] Starting experiment...")
        print(f"[rule_shift] Forward phrase: '{self.dataset.forward_phrase}'")
        print(f"[rule_shift] Reverse phrase: '{self.dataset.reverse_phrase}'")
        print(f"[rule_shift] Segment size: {self.dataset.segment_size}")
        print(f"[rule_shift] Total training: {len(self.dataset.train_bytes)} bytes")
        print(f"[rule_shift] Phase switch at byte {self.dataset.phase_switch_byte}")
        
        # 2. MANIFOLD
        grid_size = (64, 64, 64)
        dt = 0.01
        
        manifold = Manifold(
            SimulationConfig(
                dashboard=self.dashboard,
                video_path=self.video_path,
                generator=self.dataset.generate,
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                ),
                coherence=CoherenceSimulationConfig(
                    max_carriers=64,
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                    volatile_decay_mul=0.98,
                    coupling_scale=0.5,
                    grid_size=grid_size,
                    dt=dt,
                ),
            )
        )
        
        start_time = time.time()
        state = manifold.run()
        wall_time_ms = (time.time() - start_time) * 1000
        
        token_ids = state.get("token_ids")
        if token_ids is None:
            print("[rule_shift] ERROR: No token_ids in state")
            return
        
        n_particles = len(token_ids.cpu().numpy())
        print(f"[rule_shift] Tokenized {n_particles} particles")
        
        # 3. OBSERVE - run predictor
        prediction = self.predictor.observe({
            "token_ids": token_ids,
            "energies": state.get("energies", torch.ones(len(token_ids))),
            "forward_phrase": self.dataset.forward_phrase,
            "reverse_phrase": self.dataset.reverse_phrase,
            "forward_reps": self.dataset.forward_reps,
            "reverse_reps": self.dataset.reverse_reps,
            "eval_every": self.eval_every,
            "segment_size": self.dataset.segment_size,
            "phase_switch_byte": self.dataset.phase_switch_byte,
        })
        
        # Log progress
        for r in prediction.get("accuracy_history", []):
            print(f"[rule_shift] Rep {r['rep']}: {r['phase']} accuracy = {r['accuracy']:.3f} ({r['correct']}/{r['total']})")
        
        # Get mode stats
        modes = manifold.modes or {}
        amplitudes = modes.get("amplitudes")
        n_modes = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
        
        # Accumulate to inference observer
        self.inference.observe(
            state,
            manifold=manifold,
            accuracy_history=prediction.get("accuracy_history", []),
            forward_reps=self.dataset.forward_reps,
            reverse_reps=self.dataset.reverse_reps,
            segment_size=self.dataset.segment_size,
            context_length=self.context_length,
        )
        
        # Project
        self.project()
        
        # Write simulation stats
        self.write_simulation_stats(
            "rule_shift",
            n_particles=n_particles,
            n_modes=n_modes,
            n_crystallized=0,
            grid_size=grid_size,
            dt=dt,
            n_steps=1,
            wall_time_ms=wall_time_ms,
        )
        print(f"✓ Generated: paper/tables/rule_shift_stats.tex")
        
        print(f"[rule_shift] Experiment complete.")

    def observe(self, state: dict) -> dict:
        """Observer interface for compatibility."""
        return {}

    def project(self) -> dict:
        """Project observation to artifacts."""
        return self.projector.project(self.inference)
