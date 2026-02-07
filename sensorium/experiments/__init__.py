"""Experiment entrypoints (lazy imports).

We keep package import lightweight so CLI features like `--list` work even when
optional runtime dependencies for a specific experiment are unavailable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AblationsExperiment",
    "CollisionExperiment",
    "WaveTrieExperiment",
    "KernelRuleShift",
    "KernelScaling",
    "KernelCrossModal",
    "KernelImageGen",
]

_CLASS_TO_MODULE = {
    "AblationsExperiment": ".ablations",
    "CollisionExperiment": ".collision",
    "WaveTrieExperiment": ".wave_trie",
    "KernelRuleShift": ".kernel_rule_shift",
    "KernelScaling": ".kernel_scaling",
    "KernelCrossModal": ".kernel_cross_modal",
    "KernelImageGen": ".kernel_image_gen",
}


def __getattr__(name: str) -> Any:
    module_name = _CLASS_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
