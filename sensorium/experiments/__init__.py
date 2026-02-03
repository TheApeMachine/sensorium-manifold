"""Experiment entrypoints.

This package previously contained a legacy, Torch/Python semantic stack.
The *current* experiments are kernel-based (Metal on macOS) and produce
paper-ready artifacts under `paper/tables/` and `paper/figures/`.

We intentionally avoid importing legacy modules at import-time to prevent
mixing old/new implementations.
"""

from .kernel_rule_shift import KernelRuleShiftConfig, run_kernel_rule_shift
from .kernel_ablations import run_kernel_ablation_study
from .kernel_continuous import KernelContinuousConfig, run_kernel_continuous

__all__ = [
    "KernelRuleShiftConfig",
    "run_kernel_rule_shift",
    "run_kernel_ablation_study",
    "KernelContinuousConfig",
    "run_kernel_continuous",
]
