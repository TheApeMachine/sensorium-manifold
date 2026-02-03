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
from .kernel_next_token import KernelNextTokenConfig, run_kernel_next_token
from .kernel_timeseries import KernelTimeSeriesConfig, run_kernel_timeseries
from .kernel_text_diffusion import KernelTextDiffusionConfig, run_kernel_text_diffusion
from .kernel_mnist_bytes import KernelMNISTBytesConfig, run_kernel_mnist_bytes
from .kernel_image_gen import KernelImageGenConfig, run_kernel_image_gen
from .kernel_audio_gen import KernelAudioGenConfig, run_kernel_audio_gen
from .kernel_cocktail_party import KernelCocktailPartyConfig, run_kernel_cocktail_party

__all__ = [
    "KernelRuleShiftConfig",
    "run_kernel_rule_shift",
    "run_kernel_ablation_study",
    "KernelContinuousConfig",
    "run_kernel_continuous",
    "KernelNextTokenConfig",
    "run_kernel_next_token",
    "KernelTimeSeriesConfig",
    "run_kernel_timeseries",
    "KernelTextDiffusionConfig",
    "run_kernel_text_diffusion",
    "KernelMNISTBytesConfig",
    "run_kernel_mnist_bytes",
    "KernelImageGenConfig",
    "run_kernel_image_gen",
    "KernelAudioGenConfig",
    "run_kernel_audio_gen",
    "KernelCocktailPartyConfig",
    "run_kernel_cocktail_party",
]
