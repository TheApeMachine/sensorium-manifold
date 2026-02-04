"""Command-line entrypoint for running experiments.

Usage:
    python -m sensorium.experiments --experiment all
    python -m sensorium.experiments --experiment rule_shift
    python -m sensorium.experiments --experiment mnist_bytes
"""

from __future__ import annotations

import argparse
import sys

from . import (
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
    KernelScaling,
    KernelCrossModal,
    CollisionExperiment,
    MNISTTrieRecallExperiment,
)
from .image_collision import ImageCollisionExperiment


# Map experiment names to experiment classes
EXPERIMENTS = {
    "rule_shift": KernelRuleShift,
    "ablation": KernelAblations,
    "continuous": KernelContinuous,
    "next_token": KernelNextToken,
    "timeseries": KernelTimeSeries,
    "text_diffusion": KernelTextDiffusion,
    "mnist_bytes": KernelMNISTBytes,
    "image_gen": KernelImageGen,
    "audio_gen": KernelAudioGen,
    "cocktail_party": KernelCocktailParty,
    "scaling": KernelScaling,
    "cross_modal": KernelCrossModal,
    # Collision suite (concrete): tokenizer collision regime sweeps.
    "collision": CollisionExperiment,
    "image_collision": ImageCollisionExperiment,
    "mnist_trie_recall": MNISTTrieRecallExperiment,
}

# All experiments in order
ALL_EXPERIMENTS = [
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
    KernelScaling,
    KernelCrossModal,
    CollisionExperiment,
    ImageCollisionExperiment,
    MNISTTrieRecallExperiment,
]


def main():
    parser = argparse.ArgumentParser(
        description="Run kernel-based experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=f"Experiment to run. Options: all, {', '.join(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )

    args = parser.parse_args()

    if args.experiment == "all":
        experiments_to_run = ALL_EXPERIMENTS
    elif args.experiment in EXPERIMENTS:
        experiments_to_run = [EXPERIMENTS[args.experiment]]
    else:
        print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
        print(f"Available experiments: all, {', '.join(EXPERIMENTS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Run experiments
    for experiment_class in experiments_to_run:
        experiment_name = experiment_class.__name__
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        try:
            experiment = experiment_class(experiment_name=experiment_name, profile=args.profile)
            experiment.run()
            print(f"✓ Completed: {experiment_name}")
        except Exception as e:
            print(f"✗ Failed: {experiment_name}", file=sys.stderr)
            print(f"  Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Continue with other experiments even if one fails
            continue

    print(f"\n{'='*60}")
    print("All experiments completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
