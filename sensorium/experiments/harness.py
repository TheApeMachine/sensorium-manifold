"""Kernel experiment harness (Metal/MPS).

Goal: run experiments on the *new* custom-kernel implementation and generate
paper artifacts automatically under `paper/` so `paper/main.tex` picks them up.

Usage:
  - Run rule shift (writes `paper/tables/rule_shift_summary.tex` + `paper/figures/rule_shift.pdf`)
    `python3 -m sensorium.experiments.harness --experiment rule_shift`

  - Run ablations (writes `paper/tables/ablation.tex` and also refreshes rule_shift outputs)
    `python3 -m sensorium.experiments.harness --experiment ablation`

  - Run both
    `python3 -m sensorium.experiments.harness --experiment all`
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Optional

import torch

from .kernel_rule_shift import KernelRuleShiftConfig, run_kernel_rule_shift
from .kernel_ablations import run_kernel_ablation_study
from .kernel_continuous import KernelContinuousConfig, run_kernel_continuous
from .kernel_next_token import KernelNextTokenConfig, run_kernel_next_token
from .kernel_timeseries import KernelTimeSeriesConfig, run_kernel_timeseries
from .kernel_mnist_bytes import KernelMNISTBytesConfig, run_kernel_mnist_bytes
from .kernel_image_gen import KernelImageGenConfig, run_kernel_image_gen
from .kernel_audio_gen import KernelAudioGenConfig, run_kernel_audio_gen
from .kernel_text_diffusion import KernelTextDiffusionConfig, run_kernel_text_diffusion
from .kernel_cocktail_party import KernelCocktailPartyConfig, run_kernel_cocktail_party


def get_device_str() -> str:
    """Prefer Metal/MPS on macOS."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_paper_pipeline(*, device: str, run_rule_shift: bool, run_ablation: bool) -> None:
    paper_dir = Path("./paper")
    paper_dir.mkdir(parents=True, exist_ok=True)

    if run_rule_shift:
        print("[RUN] kernel rule_shift → paper/tables + paper/figures")
        cfg = KernelRuleShiftConfig(device=device)
        run_kernel_rule_shift(cfg, out_dir=paper_dir)

    if run_ablation:
        print("[RUN] kernel ablation → paper/tables/ablation.tex")
        run_kernel_ablation_study(device=device, out_dir=paper_dir, base_cfg=KernelRuleShiftConfig(device=device))


def run_kernel_experiments_all(*, device: str) -> None:
    """Run all kernel experiments and refresh paper artifacts."""
    run_paper_pipeline(device=device, run_rule_shift=True, run_ablation=True)
    # Also produce a stable kernel-physics visualization.
    run_kernel_continuous(KernelContinuousConfig(device=device, dashboard_enabled=False), out_dir=Path("./paper"))
    # Universal Tokenizer benchmarks + modality demos
    run_kernel_next_token(KernelNextTokenConfig(device=device), out_dir=Path("./paper"))
    run_kernel_timeseries(KernelTimeSeriesConfig(device=device), out_dir=Path("./paper"))
    run_kernel_text_diffusion(KernelTextDiffusionConfig(device=device), out_dir=Path("./paper"))
    # Vision/audio (may require torchvision; audio uses synthetic by default)
    run_kernel_mnist_bytes(KernelMNISTBytesConfig(device=device), out_dir=Path("./paper"))
    run_kernel_image_gen(KernelImageGenConfig(device=device), out_dir=Path("./paper"))
    run_kernel_audio_gen(KernelAudioGenConfig(device=device), out_dir=Path("./paper"))
    run_kernel_cocktail_party(KernelCocktailPartyConfig(device=device), out_dir=Path("./paper"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run kernel-based sensorium experiments (Metal/MPS)"
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "rule_shift",
            "ablation",
            "continuous",
            "next_token",
            "timeseries",
            "text_diffusion",
            "mnist_bytes",
            "image_gen",
            "audio_gen",
            "cocktail_party",
            "all",
        ],
        default="all",
        help="Which experiment to run (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (default: auto; prefers 'mps')",
    )
    
    args = parser.parse_args()

    device = args.device or get_device_str()
    print("=" * 60)
    print("SENSORIUM KERNEL EXPERIMENTS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Experiment: {args.experiment}")

    if args.experiment == "all":
        run_kernel_experiments_all(device=device)
    elif args.experiment == "continuous":
        run_kernel_continuous(KernelContinuousConfig(device=device, dashboard_enabled=False), out_dir=Path("./paper"))
    elif args.experiment == "next_token":
        run_kernel_next_token(KernelNextTokenConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "timeseries":
        run_kernel_timeseries(KernelTimeSeriesConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "text_diffusion":
        run_kernel_text_diffusion(KernelTextDiffusionConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "mnist_bytes":
        run_kernel_mnist_bytes(KernelMNISTBytesConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "image_gen":
        run_kernel_image_gen(KernelImageGenConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "audio_gen":
        run_kernel_audio_gen(KernelAudioGenConfig(device=device), out_dir=Path("./paper"))
    elif args.experiment == "cocktail_party":
        run_kernel_cocktail_party(KernelCocktailPartyConfig(device=device), out_dir=Path("./paper"))
    else:
        run_paper_pipeline(
            device=device,
            run_rule_shift=args.experiment == "rule_shift",
            run_ablation=args.experiment == "ablation",
        )


if __name__ == "__main__":
    main()
