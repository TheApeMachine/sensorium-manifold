"""Experiment suite for Thermodynamic Manifolds.

All experiments use HuggingFace datasets for real-world benchmarks.
Each experiment runs at three scales: toy, medium, full.
"""

from .base import BaseExperiment, Scale, ScaleConfig, ExperimentResult, SCALE_CONFIGS
from .timeseries import TimeSeriesExperiment, run_timeseries_experiment
from .next_token import NextTokenExperiment, run_next_token_experiment
from .image_gen import ImageGenerationExperiment, run_image_gen_experiment
from .audio_gen import AudioGenerationExperiment, run_audio_gen_experiment
from .text_diffusion import TextDiffusionExperiment, run_text_diffusion_experiment
from .mnist_bytes import MNISTBytesExperiment, run_mnist_bytes_experiment

__all__ = [
    # Base
    "BaseExperiment",
    "Scale",
    "ScaleConfig", 
    "ExperimentResult",
    "SCALE_CONFIGS",
    # Experiments
    "TimeSeriesExperiment",
    "NextTokenExperiment",
    "ImageGenerationExperiment",
    "AudioGenerationExperiment",
    "TextDiffusionExperiment",
    "MNISTBytesExperiment",
    # Runners
    "run_timeseries_experiment",
    "run_next_token_experiment",
    "run_image_gen_experiment",
    "run_audio_gen_experiment",
    "run_text_diffusion_experiment",
    "run_mnist_bytes_experiment",
]
