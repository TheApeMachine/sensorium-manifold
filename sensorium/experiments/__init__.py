"""Experiment entrypoints.

This package previously contained a legacy, Torch/Python semantic stack.
The *current* experiments are kernel-based (Metal on macOS) and produce
paper-ready artifacts under `paper/tables/` and `paper/figures/`.

We intentionally avoid importing legacy modules at import-time to prevent
mixing old/new implementations.
"""

from .ablations import AblationsExperiment
from .collision import CollisionExperiment
from .wave_trie import WaveTrieExperiment

def __main__():
    for experiment in [
        AblationsExperiment,
        CollisionExperiment,
        WaveTrieExperiment,
    ]:
        experiment(experiment_name=experiment.__name__).run()
