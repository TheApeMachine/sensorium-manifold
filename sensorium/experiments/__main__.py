"""Command-line entrypoint for running experiments.

Runs the experiments defined in the `experiments` package.
"""

from __future__ import annotations

import argparse
import os

_EXPERIMENT_NAMES = ["ablation", "collision", "wave_trie"]


def _set_interactive_matplotlib_backend() -> None:
    """Best-effort: ensure `plt.show()` actually opens a window.

    This must run before importing `matplotlib.pyplot` anywhere.
    """
    try:
        import matplotlib
    except Exception:
        return

    # If the user already forced a backend, respect it.
    if os.environ.get("MPLBACKEND"):
        return

    # Prefer native macOS backend; fall back to common interactive ones.
    for candidate in ("MacOSX", "QtAgg", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def _resolve_experiments():
    # Import lazily so we can set the matplotlib backend first.
    from . import AblationsExperiment, CollisionExperiment, WaveTrieExperiment

    experiments = {
        "ablation": AblationsExperiment,
        "collision": CollisionExperiment,
        "wave_trie": WaveTrieExperiment,
    }

    all_experiments = [
        AblationsExperiment,
        CollisionExperiment,
        WaveTrieExperiment,
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

    if bool(args.dashboard):
        # Ensure an interactive backend before any pyplot import.
        _set_interactive_matplotlib_backend()
        # Experiments use this env var for dashboard cadence.
        os.environ["THERMO_MANIFOLD_DASHBOARD_FPS"] = str(int(args.dashboard_fps))

    EXPERIMENTS, ALL_EXPERIMENTS = _resolve_experiments()

    if args.experiment == "all":
        experiments = ALL_EXPERIMENTS
    else:
        experiments = [EXPERIMENTS[args.experiment]]

    for experiment in experiments:
        try:
            exp = experiment(
                experiment_name=experiment.__name__, dashboard=bool(args.dashboard)
            )
            try:
                exp.run()
            finally:
                # Ensure dashboard recording is finalized even on errors.
                try:
                    exp.close_dashboard()
                except Exception:
                    pass
        except Exception as exc:
            if args.experiment == "all":
                print(f"[error] {experiment.__name__} failed: {exc}")
                continue
            raise


if __name__ == "__main__":
    main()
