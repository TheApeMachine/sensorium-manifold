"""Command-line entrypoint for running experiments.

Runs the experiments defined in the `experiments` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_EXPERIMENT_NAMES = [
    "collision",
]

def _resolve_experiments():
    # Import lazily so we can set the matplotlib backend first.
    from . import (
        CollisionExperiment,
    )

    experiments = {
        "collision": CollisionExperiment,
    }

    all_experiments = [
        CollisionExperiment,
    ]

    return experiments, all_experiments


def main():
    parser = argparse.ArgumentParser(prog="python -m sensorium.experiments")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", *_EXPERIMENT_NAMES],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live dashboard while running",
    )
    parser.add_argument(
        "--dashboard-fps",
        type=int,
        default=30,
        help="Dashboard refresh rate (frames per second)",
    )
    args = parser.parse_args()

    if bool(args.list):
        print("Available experiments:")
        for name in _EXPERIMENT_NAMES:
            print(f"  - {name}")
        return

    EXPERIMENTS, ALL_EXPERIMENTS = _resolve_experiments()
    repo_root = Path(__file__).resolve().parents[2]

    if args.experiment == "all":
        experiments = ALL_EXPERIMENTS
    else:
        experiments = [EXPERIMENTS[args.experiment]]

    for experiment in experiments:
        try:
            exp = experiment(
                experiment_name=experiment.__name__, dashboard=bool(args.dashboard)
            )
            exp.run()
        except Exception as exc:
            if args.experiment == "all":
                print(f"[error] {experiment.__name__} failed: {exc}")
                continue
            raise


if __name__ == "__main__":
    main()
