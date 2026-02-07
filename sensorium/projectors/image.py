"""Image generation projectors for tables and figures.

This projector consumes multi-seed `KernelImageGen` results and renders
paper-ready LaTeX + figures with uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class ImageFigureConfig:
    """Configuration for image figure."""

    name: str = "image_gen"
    format: str = "png"
    dpi: int = 300


def _ci95(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(float(n)))


def _latex_escape_text(text: str) -> str:
    out = str(text).replace("\\", "/")
    replacements = (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _frac_key(frac: float) -> str:
    return str(int(round(float(frac) * 100.0)))


def _metric_from_row(row: dict[str, Any], frac: float, key: str) -> float:
    mask_results = row.get("mask_results", {})
    if isinstance(mask_results, dict):
        cell = mask_results.get(float(frac), None)
        if isinstance(cell, dict):
            value = cell.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    flat_key = f"{key}_{_frac_key(frac)}"
    value2 = row.get(flat_key)
    if isinstance(value2, (int, float)):
        return float(value2)
    return 0.0


class ImageTableProjector(BaseProjector):
    """Projector for image generation summary table."""

    def __init__(
        self,
        output_dir: Path | None = None,
        name: str = "image_gen_summary",
        train_images: int = 100,
        test_images: int = 20,
    ):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name
        self.train_images = int(train_images)
        self.test_images = int(test_images)

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        rows = [r for r in self._get_results_list(source) if isinstance(r, dict)]
        if not rows:
            return {"status": "skipped", "reason": "no results"}

        # Pull mask levels from first row; fall back to canonical set.
        mask_levels = [0.1, 0.2, 0.3, 0.5]
        mr0 = rows[0].get("mask_results", None)
        if isinstance(mr0, dict) and mr0:
            try:
                mask_levels = sorted(float(k) for k in mr0.keys())
            except Exception:
                mask_levels = [0.1, 0.2, 0.3, 0.5]

        backend_counts: dict[str, int] = {}
        for row in rows:
            b = str(row.get("run_backend", ""))
            backend_counts[b] = int(backend_counts.get(b, 0) + 1)

        provenance = str(rows[0].get("provenance_jsonl", ""))
        provenance_tex = _latex_escape_text(provenance)
        backend_text = _latex_escape_text(
            ", ".join(f"{k}:{int(v)}" for k, v in backend_counts.items())
        )

        self.ensure_output_dir()

        ncols = int(1 + len(mask_levels))
        colspec = "l " + ("c " * len(mask_levels))
        header = " & ".join(["\\textbf{Metric}"] + [f"\\textbf{{{int(100*m)}\\% Mask}}" for m in mask_levels]) + r" \\" 

        def fmt_metric(key: str, digits: int) -> str:
            vals = []
            for frac in mask_levels:
                samples = [_metric_from_row(r, float(frac), key) for r in rows]
                mu = float(np.mean(samples)) if samples else 0.0
                ci = _ci95(samples)
                vals.append(f"{mu:.{digits}f} $\\pm$ {ci:.{digits}f}")
            return " & ".join(vals)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{MNIST inpainting via manifold-derived position-byte statistics (mean $\pm$ 95\% CI across seeds).}",
            r"\label{tab:image_gen}",
            rf"\begin{{tabular}}{{{colspec.strip()}}}",
            r"\toprule",
            header,
            r"\midrule",
            f"PSNR (dB) & {fmt_metric('psnr', 2)} \\\\",
            f"MAE (pixels) & {fmt_metric('mae', 2)} \\\\",
            f"MSE & {fmt_metric('mse', 1)} \\\\",
            r"\midrule",
            rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{Dataset}}}} \\",
            rf"\multicolumn{{{ncols}}}{{l}}{{\quad Training images: {int(rows[0].get('train_images', self.train_images))}}} \\",
            rf"\multicolumn{{{ncols}}}{{l}}{{\quad Test images: {int(rows[0].get('test_images', self.test_images))}}} \\",
            rf"\multicolumn{{{ncols}}}{{l}}{{\quad Image size: 28$\times$28 = 784 pixels}} \\",
            rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{Backend counts: {backend_text}}}}} \\",
            rf"\multicolumn{{{ncols}}}{{l}}{{\texttt{{{provenance_tex}}}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {"status": "success", "path": str(output_path)}


class ImageFigureProjector(BaseProjector):
    """Projector for image generation 3-panel figure."""

    def __init__(
        self,
        config: ImageFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        self.config = config or ImageFigureConfig(**kwargs)

    @staticmethod
    def _representative_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Prefer the run with best PSNR at 30% mask for panel A examples.
        def score(row: dict[str, Any]) -> float:
            return float(_metric_from_row(row, 0.3, "psnr"))

        return max(rows, key=score)

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [r for r in self._get_results_list(source) if isinstance(r, dict)]
        if not rows:
            return {"status": "skipped", "reason": "no results"}

        mask_levels = [0.1, 0.2, 0.3, 0.5]
        mr0 = rows[0].get("mask_results", None)
        if isinstance(mr0, dict) and mr0:
            try:
                mask_levels = sorted(float(k) for k in mr0.keys())
            except Exception:
                mask_levels = [0.1, 0.2, 0.3, 0.5]

        rep = self._representative_row(rows)
        examples = rep.get("examples", [])
        if not isinstance(examples, list):
            examples = []

        self.ensure_output_dir()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Panel A: Example reconstructions at 30% mask.
        ax = axes[0]
        target_mask = 0.3
        examples_at_level = [
            e for e in examples if abs(float(e.get("mask_frac", -1.0)) - target_mask) < 1e-9
        ]
        if len(examples_at_level) < 4:
            examples_at_level = examples[:4]
        n_examples = min(4, len(examples_at_level))

        if n_examples > 0:
            composite = np.ones((28 * 2, 28 * n_examples + max(0, n_examples - 1), 3)) * 0.9
            for i, example in enumerate(examples_at_level[:n_examples]):
                original = np.frombuffer(bytes(example.get("original", b"")), dtype=np.uint8)
                recon = np.frombuffer(bytes(example.get("reconstructed", b"")), dtype=np.uint8)
                if original.size != 784 or recon.size != 784:
                    continue
                original_img = original.reshape(28, 28).astype(np.float64)
                recon_img = recon.reshape(28, 28).astype(np.float64)
                x_offset = i * 29
                v0 = original_img / 255.0
                v1 = recon_img / 255.0
                for c in range(3):
                    composite[:28, x_offset : x_offset + 28, c] = v0
                    composite[28:56, x_offset : x_offset + 28, c] = v1

            ax.imshow(composite)
            ax.set_xticks([14 + i * 29 for i in range(n_examples)])
            labels = [f"d={int(examples_at_level[i].get('label', -1))}" for i in range(n_examples)]
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_xlabel(f"Digit (at {int(target_mask*100)}% mask)", fontsize=10)
            ax.set_yticks([14, 42])
            ax.set_yticklabels(["Original", "Reconstructed"], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.02, 0.98, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

        # Panel B: PSNR vs mask fraction (+ MAE on secondary axis) with CI.
        ax = axes[1]
        x = np.asarray([100.0 * float(m) for m in mask_levels], dtype=np.float64)
        psnr_mu: list[float] = []
        psnr_ci: list[float] = []
        mae_mu: list[float] = []
        mae_ci: list[float] = []
        for frac in mask_levels:
            pvals = [_metric_from_row(r, float(frac), "psnr") for r in rows]
            mvals = [_metric_from_row(r, float(frac), "mae") for r in rows]
            psnr_mu.append(float(np.mean(pvals)) if pvals else 0.0)
            psnr_ci.append(_ci95(pvals))
            mae_mu.append(float(np.mean(mvals)) if mvals else 0.0)
            mae_ci.append(_ci95(mvals))

        ax.errorbar(x, psnr_mu, yerr=psnr_ci, marker="o", linewidth=2.0, capsize=3, label="PSNR")
        ax.set_xlabel("Mask fraction (%)", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10)

        ax2 = ax.twinx()
        ax2.errorbar(x, mae_mu, yerr=mae_ci, marker="s", linestyle="--", linewidth=2.0, capsize=3, label="MAE")
        ax2.set_ylabel("MAE (pixel intensity)", fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.text(0.02, 0.98, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

        # Panel C: MAE breakdown by mask level (mean ± CI).
        ax = axes[2]
        x_pos = np.arange(len(mask_levels), dtype=np.float64)
        bars = ax.bar(x_pos, mae_mu, yerr=mae_ci, capsize=3)
        for bar, mae in zip(bars, mae_mu):
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                float(bar.get_height() + 0.3),
                f"{mae:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(m*100)}%" for m in mask_levels], fontsize=10)
        ax.set_xlabel("Mask fraction", fontsize=10)
        ax.set_ylabel("Mean Absolute Error", fontsize=10)
        ymax = max(mae_mu) if mae_mu else 1.0
        ax.set_ylim(0.0, float(max(1.0, ymax * 1.25)))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.02, 0.98, "C", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        return {"status": "success", "path": str(output_path)}
