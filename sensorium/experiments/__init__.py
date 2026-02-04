"""Experiment entrypoints.

This package previously contained a legacy, Torch/Python semantic stack.
The *current* experiments are kernel-based (Metal on macOS) and produce
paper-ready artifacts under `paper/tables/` and `paper/figures/`.

We intentionally avoid importing legacy modules at import-time to prevent
mixing old/new implementations.
"""

from .kernel_rule_shift import KernelRuleShift
from .kernel_ablations import KernelAblations
from .kernel_continuous import KernelContinuous
from .kernel_next_token import KernelNextToken
from .kernel_timeseries import KernelTimeSeries
from .kernel_text_diffusion import KernelTextDiffusion
from .kernel_mnist_bytes import KernelMNISTBytes
from .kernel_image_gen import KernelImageGen
from .kernel_audio_gen import KernelAudioGen
from .kernel_cocktail_party import KernelCocktailParty
from .collision import CollisionExperiment
from .image_collision import ImageCollisionExperiment
from .mnist_trie_recall import MNISTTrieRecallExperiment


def __main__():
    for experiment in [
        KernelRuleShift,
        KernelAblations,
        KernelContinuous,
        KernelNextToken,
        KernelTimeSeries,
        KernelTextDiffusion,
        KernelMNISTBytes,
        KernelImageGen,
        KernelAudioGen,
        KernelCocktailParty,
        CollisionExperiment,
        MNISTTrieRecallExperiment,
    ]:
        experiment(experiment_name=experiment.__name__).run()
