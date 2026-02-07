"""Cross-modal projectors for real-manifold experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class CrossModalFigureConfig:
    """Configuration for cross-modal figures."""

    name: str = "cross_modal"
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


class CrossModalFigureProjector(BaseProjector):
    """Generate cross-modal figures from run rows."""

    def __init__(
        self,
        config: CrossModalFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        self.config = config or CrossModalFigureConfig(**kwargs)

    @staticmethod
    def _group_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("scenario", ""))
            grouped.setdefault(key, []).append(row)
        return grouped

    @staticmethod
    def _representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped = CrossModalFigureProjector._group_by_scenario(rows)
        out: list[dict[str, Any]] = []
        for key in sorted(grouped.keys()):
            bucket = grouped[key]
            bucket2 = sorted(bucket, key=lambda r: float(r.get("mse", 1e9)))
            out.append(bucket2[0])
        return out

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        rows = [r for r in self._get_results_list(source) if isinstance(r, dict)]
        if not rows:
            return {"status": "skipped", "reason": "no rows"}

        reps = self._representative_rows(rows)
        grouped = self._group_by_scenario(rows)

        self.ensure_output_dir()

        # Figure 1: Original/reconstruction + summary metrics.
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)

        for i, row in enumerate(reps[:4]):
            ax = fig.add_subplot(gs[0, i])
            original = np.asarray(row.get("original", np.zeros((32, 32))), dtype=np.float64)
            ax.imshow(original, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(str(row.get("scenario", "")))
            ax.axis("off")

        for i, row in enumerate(reps[:4]):
            ax = fig.add_subplot(gs[1, i])
            recon = np.asarray(row.get("reconstructed", np.zeros((32, 32))), dtype=np.float64)
            ax.imshow(recon, cmap="viridis", vmin=0.0, vmax=1.0)
            mse = float(row.get("mse", 0.0))
            psnr = float(row.get("psnr", 0.0))
            ax.set_title(f"MSE={mse:.4f}, PSNR={psnr:.2f} dB", fontsize=9)
            ax.axis("off")

        # Panel A: Frequency particles (representative row).
        ax = fig.add_subplot(gs[2, 0:2])
        anchor = reps[0]
        fu = np.asarray(anchor.get("freq_u", []), dtype=np.float64)
        fv = np.asarray(anchor.get("freq_v", []), dtype=np.float64)
        fe = np.asarray(anchor.get("freq_energy", []), dtype=np.float64)
        if fu.size and fv.size and fe.size:
            emax = float(np.max(fe)) if fe.size else 0.0
            sizes = np.clip((fe / max(emax, 1e-9)) * 160.0, 12.0, 180.0)
            sc = ax.scatter(fv, fu, c=fe, s=sizes, cmap="plasma", alpha=0.82)
            fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.01, label="Energy")
        ax.axhline(0.0, linewidth=0.6)
        ax.axvline(0.0, linewidth=0.6)
        ax.set_xlabel("v (horizontal frequency)")
        ax.set_ylabel("u (vertical frequency)")
        ax.set_title("A  Frequency particles (representative run)")
        ax.grid(True, alpha=0.2)

        # Panel B: Per-scenario MSE with CI + PSNR trend.
        ax = fig.add_subplot(gs[2, 2])
        names = []
        mse_mu = []
        mse_ci = []
        psnr_mu = []
        for name in sorted(grouped.keys()):
            names.append(name)
            bucket = grouped[name]
            mses = [float(r.get("mse", 0.0)) for r in bucket]
            psnrs = [float(r.get("psnr", 0.0)) for r in bucket]
            mse_mu.append(float(np.mean(mses)) if mses else 0.0)
            mse_ci.append(_ci95(mses))
            psnr_mu.append(float(np.mean(psnrs)) if psnrs else 0.0)
        x = np.arange(len(names), dtype=np.float64)
        ax.bar(x, mse_mu, yerr=mse_ci, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MSE")
        ax.set_title("B  Reconstruction error (mean ± 95% CI)")
        ax.grid(True, axis="y", alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(x, psnr_mu, marker="o")
        ax2.set_ylabel("PSNR (dB)")

        # Panel C: text centroid coverage and distance.
        ax = fig.add_subplot(gs[2, 3])
        coverage_mu = []
        dist_mu = []
        for name in sorted(grouped.keys()):
            bucket = grouped[name]
            covs = [float(r.get("text_label_coverage", 0.0)) for r in bucket]
            dists = [float(r.get("image_text_centroid_dist", 0.0)) for r in bucket]
            coverage_mu.append(float(np.mean(covs)) if covs else 0.0)
            dist_mu.append(float(np.mean(dists)) if dists else 0.0)
        ax.bar(x, coverage_mu)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title("C  Text centroid coverage")
        ax.grid(True, axis="y", alpha=0.2)
        ax3 = ax.twinx()
        ax3.plot(x, dist_mu, marker="s")
        ax3.set_ylabel("Centroid distance")

        plt.tight_layout()
        p0 = self.output_dir / f"{self.config.name}.{self.config.format}"
        fig.savefig(p0, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        # Figure 2: 3D embedding from actual manifold positions.
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        anchor = reps[0]
        img_pts = np.asarray(anchor.get("image_points", []), dtype=np.float64)
        img_en = np.asarray(anchor.get("image_point_energy", []), dtype=np.float64)
        txt_pts = np.asarray(anchor.get("text_points", []), dtype=np.float64)
        if img_pts.ndim == 2 and img_pts.shape[0] > 0:
            if img_en.size == img_pts.shape[0]:
                sc = ax.scatter(
                    img_pts[:, 0],
                    img_pts[:, 1],
                    img_pts[:, 2],
                    c=img_en,
                    cmap="Blues",
                    s=20,
                    alpha=0.45,
                    label="Image particles",
                )
                fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="Energy")
            else:
                ax.scatter(
                    img_pts[:, 0],
                    img_pts[:, 1],
                    img_pts[:, 2],
                    s=20,
                    alpha=0.45,
                    label="Image particles",
                )
        if txt_pts.ndim == 2 and txt_pts.shape[0] > 0:
            ax.scatter(
                txt_pts[:, 0],
                txt_pts[:, 1],
                txt_pts[:, 2],
                s=28,
                alpha=0.85,
                marker="^",
                color="tab:red",
                label="Text particles",
            )

        for item in anchor.get("text_label_centroids", []):
            if not isinstance(item, dict):
                continue
            x0 = float(item.get("x", 0.0))
            y0 = float(item.get("y", 0.0))
            z0 = float(item.get("z", 0.0))
            label = str(item.get("label", ""))
            ax.scatter([x0], [y0], [z0], marker="*", s=240, color="red", edgecolors="black")
            ax.text(x0, y0, z0, label)

        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        ax.set_zlabel("Dim 2")
        ax.set_title("Cross-modal manifold embedding (actual state)")
        ax.legend(loc="upper left")

        p1 = self.output_dir / "cross_modal_embedding.png"
        fig.savefig(p1, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: Aggregate 2D frequency-energy scatter across representative runs.
        fig, ax = plt.subplots(figsize=(10, 8))
        all_u = []
        all_v = []
        all_e = []
        for row in reps:
            all_u.extend(np.asarray(row.get("freq_u", []), dtype=np.float64).tolist())
            all_v.extend(np.asarray(row.get("freq_v", []), dtype=np.float64).tolist())
            all_e.extend(np.asarray(row.get("freq_energy", []), dtype=np.float64).tolist())
        if all_u and all_v and all_e:
            u = np.asarray(all_u, dtype=np.float64)
            v = np.asarray(all_v, dtype=np.float64)
            e = np.asarray(all_e, dtype=np.float64)
            emax = float(np.max(e)) if e.size else 0.0
            sizes = np.clip((e / max(emax, 1e-9)) * 170.0, 10.0, 180.0)
            sc = ax.scatter(v, u, c=e, s=sizes, cmap="plasma", alpha=0.82)
            fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02, label="Energy")
        ax.axhline(0.0, linewidth=0.6)
        ax.axvline(0.0, linewidth=0.6)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("v (horizontal frequency)")
        ax.set_ylabel("u (vertical frequency)")
        ax.set_title("Frequency-space energy distribution")

        p2 = self.output_dir / "frequency_particles.png"
        fig.savefig(p2, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "success",
            "paths": [str(p0), str(p1), str(p2)],
        }


class CrossModalTableProjector(BaseProjector):
    """Generate LaTeX summary table with seed uncertainty."""

    def __init__(self, output_dir: Path | None = None, name: str = "cross_modal_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        rows = [r for r in self._get_results_list(source) if isinstance(r, dict)]
        if not rows:
            return {"status": "skipped", "reason": "no rows"}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            scenario = str(row.get("scenario", ""))
            grouped.setdefault(scenario, []).append(row)

        backend_counts: dict[str, int] = {}
        for row in rows:
            b = str(row.get("run_backend", ""))
            backend_counts[b] = int(backend_counts.get(b, 0) + 1)

        provenance = str(rows[0].get("provenance_jsonl", ""))
        provenance_tex = _latex_escape_text(provenance)

        self.ensure_output_dir()

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-modal reconstruction and alignment metrics (mean $\pm$ 95\% CI across seeds).}",
            r"\label{tab:cross_modal}",
            r"\begin{tabular}{l c c c c c}",
            r"\toprule",
            r"\textbf{Scenario} & \textbf{MSE} & \textbf{PSNR (dB)} & \textbf{Text Cov.} & \textbf{Centroid Dist.} & \textbf{Mass Split (I/T)} \\",
            r"\midrule",
        ]

        def pm(mu: float, ci: float, digits: int = 3) -> str:
            return f"{mu:.{digits}f} $\\pm$ {ci:.{digits}f}"

        for scenario in sorted(grouped.keys()):
            bucket = grouped[scenario]
            mses = [float(r.get("mse", 0.0)) for r in bucket]
            psnrs = [float(r.get("psnr", 0.0)) for r in bucket]
            covs = [float(r.get("text_label_coverage", 0.0)) for r in bucket]
            dists = [float(r.get("image_text_centroid_dist", 0.0)) for r in bucket]
            imass = [float(r.get("image_mass_share", 0.0)) for r in bucket]
            tmass = [float(r.get("text_mass_share", 0.0)) for r in bucket]

            scenario_tex = _latex_escape_text(scenario.replace("_", " "))
            row_tex = (
                f"{scenario_tex}"
                f" & {pm(float(np.mean(mses)) if mses else 0.0, _ci95(mses), 4)}"
                f" & {pm(float(np.mean(psnrs)) if psnrs else 0.0, _ci95(psnrs), 2)}"
                f" & {pm(float(np.mean(covs)) if covs else 0.0, _ci95(covs), 3)}"
                f" & {pm(float(np.mean(dists)) if dists else 0.0, _ci95(dists), 3)}"
                f" & {float(np.mean(imass)) if imass else 0.0:.2f}/{float(np.mean(tmass)) if tmass else 0.0:.2f} \\\\\n"
            )
            lines.append(row_tex.rstrip("\n"))

        backend_text = _latex_escape_text(
            ", ".join(f"{k}:{int(v)}" for k, v in backend_counts.items())
        )
        lines.extend(
            [
                r"\midrule",
                rf"\multicolumn{{6}}{{l}}{{\textit{{Backend counts: {backend_text}}}}} \\",
                rf"\multicolumn{{6}}{{l}}{{\texttt{{{provenance_tex}}}}} \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        out = self.output_dir / f"{self.name}.tex"
        out.write_text("\n".join(lines), encoding="utf-8")
        return {"status": "success", "path": str(out)}
