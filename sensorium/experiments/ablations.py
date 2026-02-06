"""Ablation study.

Tests the contribution of individual components by disabling them.
"""

from __future__ import annotations

from .base import Experiment

class AblationsExperiment(Experiment):
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
        dashboard: bool = False,
    ):
        super().__init__(experiment_name, profile, dashboard=dashboard)

    def observe(self, state: dict):
        # Clear, easy-to-spot terminal health mark.
        print(f"[ablations] {self.terminal_health(state)}", flush=True)

    def run(self):
        pass