"""
Experiment Harness

Runs all experiments at specified scales and generates paper artifacts.

Usage:
    # Run all experiments at toy scale
    python -m thermo_manifold.experiments.harness --scale toy
    
    # Run specific experiment
    python -m thermo_manifold.experiments.harness --experiment next_token --scale medium
    
    # Run all at all scales (warning: slow!)
    python -m thermo_manifold.experiments.harness --scale all
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base import Scale, ExperimentResult
from .timeseries import TimeSeriesExperiment
from .next_token import NextTokenExperiment
from .image_gen import ImageGenerationExperiment
from .audio_gen import AudioGenerationExperiment
from .text_diffusion import TextDiffusionExperiment


# Experiment registry
EXPERIMENTS = {
    "timeseries": TimeSeriesExperiment,
    "next_token": NextTokenExperiment,
    "image_gen": ImageGenerationExperiment,
    "audio_gen": AudioGenerationExperiment,
    "text_diffusion": TextDiffusionExperiment,
}


def get_device() -> torch.device:
    """Get the best available device.
    
    Note: torch_scatter doesn't support MPS, so we fall back to CPU
    on Apple Silicon until scatter operations are reimplemented.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is fine as long as we aren't using torch_scatter.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            import torch_scatter  # type: ignore  # noqa: F401

            has_torch_scatter = True
        except Exception:
            has_torch_scatter = False
        if not has_torch_scatter:
            return torch.device("mps")
    return torch.device("cpu")


def run_single_experiment(
    name: str,
    scale: Scale,
    device: torch.device,
    *,
    profile_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run a single experiment."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    
    exp_class = EXPERIMENTS[name]
    experiment = exp_class(scale=scale, device=device)

    if profile_dir is None:
        return experiment.run()

    # Profile even if the experiment fails; always write stats in a finally.
    import cProfile
    import pstats
    import io

    profile_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{scale.value}"
    prof_path = profile_dir / f"{stem}.prof"
    txt_path = profile_dir / f"{stem}.txt"

    pr = cProfile.Profile()
    pr.enable()
    try:
        return experiment.run()
    finally:
        pr.disable()
        pr.dump_stats(str(prof_path))
        # Also write a readable summary for quick inspection.
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
        txt_path.write_text(s.getvalue(), encoding="utf-8")


def generate_latex_table(results: List[ExperimentResult]) -> str:
    """Generate LaTeX summary table from results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experiment results across scales. Success indicates the experiment completed without errors.}",
        r"\label{tab:experiments}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Experiment} & \textbf{Scale} & \textbf{Success} & \textbf{Key Metric} & \textbf{Value} \\",
        r"\midrule",
    ]
    
    for result in results:
        status = r"\checkmark" if result.success else r"$\times$"
        
        # Get the most relevant metric
        if "final" in result.metrics:
            final = result.metrics["final"]
            if "accuracy" in final:
                key_metric = "Accuracy"
                value = f"{final['accuracy']:.2%}"
            elif "mse" in final:
                key_metric = "MSE"
                value = f"{final['mse']:.4f}"
            elif "reconstruction_mse" in final:
                key_metric = "Recon. MSE"
                value = f"{final['reconstruction_mse']:.4f}"
            elif "perplexity" in final:
                key_metric = "Perplexity"
                value = f"{final['perplexity']:.2f}"
            else:
                key_metric = "--"
                value = "--"
        else:
            key_metric = "--"
            value = "--"
        
        lines.append(
            f"{result.name} & {result.scale} & {status} & {key_metric} & {value} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_report(results: List[ExperimentResult], output_dir: Path) -> None:
    """Generate full experiment report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [
            {
                "name": r.name,
                "scale": r.scale,
                "success": r.success,
                "failure_reason": r.failure_reason,
                "metrics": {
                    k: v for k, v in r.metrics.items() 
                    if k == "final" or not isinstance(v, (list, dict))
                },
            }
            for r in results
        ],
    }
    
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # LaTeX table
    latex_table = generate_latex_table(results)
    with open(output_dir / "experiment_results.tex", "w") as f:
        f.write(latex_table)
    
    # Markdown summary
    md_lines = [
        "# Experiment Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Experiment | Scale | Success | Key Metric |",
        "|------------|-------|---------|------------|",
    ]
    
    for r in results:
        status = "✓" if r.success else f"✗ ({r.failure_reason})"
        
        if "final" in r.metrics:
            final = r.metrics["final"]
            if "accuracy" in final:
                metric = f"Acc: {final['accuracy']:.2%}"
            elif "mse" in final:
                metric = f"MSE: {final['mse']:.4f}"
            elif "reconstruction_mse" in final:
                metric = f"MSE: {final['reconstruction_mse']:.4f}"
            else:
                metric = "--"
        else:
            metric = "--"
        
        md_lines.append(f"| {r.name} | {r.scale} | {status} | {metric} |")
    
    md_lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    for r in results:
        md_lines.append(f"### {r.name} ({r.scale})")
        md_lines.append("")
        if r.success and "final" in r.metrics:
            for k, v in r.metrics["final"].items():
                md_lines.append(f"- **{k}**: {v}")
        elif r.failure_reason:
            md_lines.append(f"**Failed**: {r.failure_reason}")
        md_lines.append("")
    
    with open(output_dir / "experiment_results.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"\nReports saved to {output_dir}")


def run_all_experiments(
    scales: List[Scale],
    experiments: Optional[List[str]] = None,
    output_dir: Path = Path("./artifacts/experiments"),
    *,
    profile: bool = False,
) -> List[ExperimentResult]:
    """Run all (or selected) experiments at specified scales."""
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Scales: {[s.value for s in scales]}")
    print(f"Experiments: {experiments or 'all'}")
    print("=" * 60)
    
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    results: List[ExperimentResult] = []
    profile_dir = (output_dir / "profiles") if profile else None
    
    total = len(experiments) * len(scales)
    current = 0
    
    for exp_name in experiments:
        for scale in scales:
            current += 1
            print(f"\n[{current}/{total}] Running {exp_name} at {scale.value} scale...")
            
            try:
                result = run_single_experiment(exp_name, scale, device, profile_dir=profile_dir)
                results.append(result)
                
                if result.success:
                    print(f"    ✓ Success")
                else:
                    print(f"    ✗ Failed: {result.failure_reason}")
                    
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                results.append(ExperimentResult(
                    name=exp_name,
                    scale=scale.value,
                    goal="",
                    success=False,
                    metrics={},
                    failure_reason=str(e),
                ))
    
    # Generate reports
    generate_report(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    
    print(f"Total: {len(results)}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    
    if failures > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r.success:
                print(f"  - {r.name} ({r.scale}): {r.failure_reason}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thermodynamic manifold experiments"
    )
    parser.add_argument(
        "--scale",
        choices=["toy", "medium", "full", "all"],
        default="toy",
        help="Scale to run experiments at (default: toy)",
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write cProfile stats per experiment to <output-dir>/profiles/",
    )
    
    args = parser.parse_args()
    
    # Parse scales
    if args.scale == "all":
        scales = [Scale.TOY, Scale.MEDIUM, Scale.FULL]
    else:
        scales = [Scale(args.scale)]
    
    # Parse experiments
    if args.experiment == "all":
        experiments = None  # Will run all
    else:
        experiments = [args.experiment]
    
    print("=" * 60)
    print("THERMODYNAMIC MANIFOLD - EXPERIMENT HARNESS")
    print("=" * 60)
    
    run_all_experiments(
        scales=scales,
        experiments=experiments,
        output_dir=Path(args.output_dir),
        profile=bool(args.profile),
    )


if __name__ == "__main__":
    main()
