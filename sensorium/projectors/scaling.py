"""Scaling projectors for transparent table and figure evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class ScalingFigureConfig:
    name: str = "scaling"
    format: str = "png"
    dpi: int = 300


def _pm(mu: float, sd: float, *, percent: bool = False) -> str:
    if percent:
        return f"{100.0 * mu:.1f}\\% $\\pm$ {100.0 * sd:.1f}\\%"
    return f"{mu:.3f} $\\pm$ {sd:.3f}"


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


class ScalingTableProjector(BaseProjector):
    def __init__(self, output_dir: Path | None = None, name: str = "scaling_summary"):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        results = self._get_results_list(source)
        if not results:
            return {"status": "skipped", "reason": "no results"}

        row = results[0]
        pop = row.get("population", {})
        inter = row.get("interference", {})
        latency = row.get("latency", {})
        mode_scaling = row.get("mode_scaling", {})
        gen = row.get("generalization", {})
        cond = row.get("conditions", {})
        backend_counts = row.get("backend_counts", {})
        provenance = str(row.get("provenance_jsonl", ""))
        provenance_tex = _latex_escape_text(provenance)

        self.ensure_output_dir()
        br = r"\\"
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Scaling evidence with explicit uncertainty and conditions (mean $\pm$ std over seeds unless stated).}",
            r"\label{tab:scaling}",
            r"\begin{tabular}{l l}",
            r"\toprule",
            rf"\textbf{{Check}} & \textbf{{Evidence}} {br}",
            r"\midrule",
            rf"\multicolumn{{2}}{{l}}{{\textit{{Experimental Conditions}}}} {br}",
            rf"\quad Seeds & {', '.join(str(int(s)) for s in cond.get('seeds', []))} {br}",
            rf"\quad Particle sweep & {', '.join(str(int(v)) for v in cond.get('particle_counts', []))} {br}",
            rf"\quad Mode-count sweep ($k$) & {', '.join(str(int(v)) for v in cond.get('mode_counts', []))} {br}",
            rf"\quad Sequence sweep & {', '.join(str(int(v)) for v in cond.get('sequence_lengths', []))} {br}",
            rf"\quad Backend counts & {', '.join(f'{k}:{int(v)}' for k, v in backend_counts.items())} {br}",
            rf"\quad Provenance log & \texttt{{{provenance_tex}}} {br}",
            r"\midrule",
            rf"\multicolumn{{2}}{{l}}{{\textit{{$\omega$-Field Dynamics}}}} {br}",
            rf"\quad Final active modes & {float(pop.get('n_modes_final_mean', 0.0)):.2f} {br}",
            rf"\quad Final crystallized modes & {float(pop.get('n_crystallized_final_mean', 0.0)):.2f} {br}",
            rf"\quad Pruning rate & {float(pop.get('pruning_rate_mean', 0.0)):.3f} {br}",
            rf"\quad Pruning rate 95\% CI & $\pm$ {float(pop.get('pruning_rate_ci95', 0.0)):.3f} {br}",
            r"\midrule",
            rf"\multicolumn{{2}}{{l}}{{\textit{{Latency Scaling (fixed $k$, tested range)}}}} {br}",
            rf"\quad Power-law exponent $\alpha$ in $t_\mathrm{{step}}(N)=aN^\alpha$ & {float(latency.get('fit_alpha', 0.0)):.3f} {br}",
            rf"\quad 95\% CI for $\alpha$ & [{float(latency.get('fit_alpha_ci_low', 0.0)):.3f}, {float(latency.get('fit_alpha_ci_high', 0.0)):.3f}] {br}",
            rf"\quad Mean coefficient of variation (ms/step) & {100.0 * float(latency.get('cv_mean', 0.0)):.1f}\% {br}",
            r"\midrule",
            rf"\multicolumn{{2}}{{l}}{{\textit{{Mode-Count Scaling (fixed grid)}}}} {br}",
        ]

        mrows = list(mode_scaling.get("rows", []))
        if mrows:
            first = mrows[0]
            last = mrows[-1]
            lines.append(
                rf"\quad ms/step at $k={int(first.get('omega_num_modes', 0))}$ & {_pm(float(first.get('ms_per_step_mean', 0.0)), float(first.get('ms_per_step_std', 0.0)))} {br}"
            )
            lines.append(
                rf"\quad ms/step at $k={int(last.get('omega_num_modes', 0))}$ & {_pm(float(last.get('ms_per_step_mean', 0.0)), float(last.get('ms_per_step_std', 0.0)))} {br}"
            )
            lines.append(
                rf"\quad Power-law exponent $\beta$ in $t_\mathrm{{step}}(k)=bk^\beta$ & {float(mode_scaling.get('fit_alpha', 0.0)):.3f} {br}"
            )

        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{2}}{{l}}{{\textit{{Interference at Scale}}}} {br}")

        irows = list(inter.get("rows", []))
        if irows:
            last = irows[-1]
            lines.append(
                rf"\quad Efficiency at highest pattern count ({int(last.get('n_patterns', 0))}) & {_pm(float(last.get('efficiency_mean', 0.0)), float(last.get('efficiency_std', 0.0)), percent=True)} {br}"
            )
            lines.append(
                rf"\quad Conflict score at highest pattern count & {_pm(float(last.get('conflict_mean', 0.0)), float(last.get('conflict_std', 0.0)))} {br}"
            )

        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{2}}{{l}}{{\textit{{Generalization}}}} {br}")
        for g in gen.get("rows", []):
            gname = str(g.get("name", "")).replace("_", " ").title()
            lines.append(
                rf"\quad {gname} structure ratio & {_pm(float(g.get('structure_ratio_mean', 0.0)), float(g.get('structure_ratio_std', 0.0)), percent=True)} {br}"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        out = self.output_dir / f"{self.name}.tex"
        out.write_text("\n".join(lines))
        return {"status": "success", "path": str(out)}


class ScalingDynamicsFigureProjector(BaseProjector):
    def __init__(
        self,
        config: ScalingFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        self.config = config or ScalingFigureConfig(**kwargs)

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = self._get_results_list(source)
        if not results:
            return {"status": "skipped", "reason": "no results"}

        row = results[0]
        pop = row.get("population", {})
        hist_m = pop.get("history_mean", {})
        hist_s = pop.get("history_std", {})
        inter_rows = row.get("interference", {}).get("rows", [])
        gen_rows = row.get("generalization", {}).get("rows", [])

        self.ensure_output_dir()
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        steps = np.asarray(hist_m.get("step", []), dtype=np.float64)
        n_modes = np.asarray(hist_m.get("n_modes", []), dtype=np.float64)
        n_modes_sd = np.asarray(hist_s.get("n_modes", []), dtype=np.float64)
        cryst = np.asarray(hist_m.get("n_crystallized", []), dtype=np.float64)
        cryst_sd = np.asarray(hist_s.get("n_crystallized", []), dtype=np.float64)
        if steps.size:
            ax.plot(
                steps, n_modes, color="#4C78A8", linewidth=2.0, label="Active modes"
            )
            ax.fill_between(
                steps,
                n_modes - n_modes_sd,
                n_modes + n_modes_sd,
                color="#4C78A8",
                alpha=0.18,
            )
            ax.plot(steps, cryst, color="#59A14F", linewidth=2.0, label="Crystallized")
            ax.fill_between(
                steps, cryst - cryst_sd, cryst + cryst_sd, color="#59A14F", alpha=0.18
            )
            ax.legend(fontsize=8)
        ax.set_title("Mode dynamics (mean ± std)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        ax = axes[0, 1]
        births = np.asarray(hist_m.get("n_births", []), dtype=np.float64)
        births_sd = np.asarray(hist_s.get("n_births", []), dtype=np.float64)
        deaths = np.asarray(hist_m.get("n_deaths", []), dtype=np.float64)
        deaths_sd = np.asarray(hist_s.get("n_deaths", []), dtype=np.float64)
        if steps.size:
            ax.plot(steps, births, color="#76B7B2", linewidth=2.0, label="Births")
            ax.fill_between(
                steps,
                births - births_sd,
                births + births_sd,
                color="#76B7B2",
                alpha=0.18,
            )
            ax.plot(steps, deaths, color="#E15759", linewidth=2.0, label="Deaths")
            ax.fill_between(
                steps,
                deaths - deaths_sd,
                deaths + deaths_sd,
                color="#E15759",
                alpha=0.18,
            )
            ax.legend(fontsize=8)
        ax.set_title("Birth/death dynamics")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count per step")
        ax.grid(True, alpha=0.2)

        ax = axes[1, 0]
        if inter_rows:
            x = np.asarray([r["n_patterns"] for r in inter_rows], dtype=np.float64)
            y = np.asarray([r["efficiency_mean"] for r in inter_rows], dtype=np.float64)
            ysd = np.asarray(
                [r["efficiency_std"] for r in inter_rows], dtype=np.float64
            )
            ax.errorbar(
                x, y, yerr=ysd, marker="o", linewidth=2.0, color="#F28E2B", capsize=3
            )
            ax.set_xscale("log", base=2)
        ax.set_title("Interference efficiency")
        ax.set_xlabel("Pattern count")
        ax.set_ylabel("Crystallized / patterns")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2)

        ax = axes[1, 1]
        if gen_rows:
            names = [str(r["name"]).replace("_", "\n") for r in gen_rows]
            mu = [float(r["structure_ratio_mean"]) for r in gen_rows]
            sd = [float(r["structure_ratio_std"]) for r in gen_rows]
            x = np.arange(len(names), dtype=np.float64)
            ax.bar(x, mu, yerr=sd, color="#B07AA1", capsize=3, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=8)
        ax.set_title("Structure ratio by data type")
        ax.set_ylabel("Crystallized / 64")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / f"scaling_dynamics.{self.config.format}"
        fig.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return {"status": "success", "path": str(out)}


class ScalingComputeFigureProjector(BaseProjector):
    def __init__(
        self,
        config: ScalingFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        self.config = config or ScalingFigureConfig(**kwargs)

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = self._get_results_list(source)
        if not results:
            return {"status": "skipped", "reason": "no results"}

        row = results[0]
        compute = row.get("compute", {})
        latency = row.get("latency", {})
        p_rows = compute.get("by_particles", [])
        g_rows = compute.get("by_grid", [])
        l_rows = latency.get("rows", [])

        self.ensure_output_dir()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        ax = axes[0]
        if p_rows:
            x = np.asarray([r["n_particles"] for r in p_rows], dtype=np.float64)
            y = np.asarray([r["ms_per_step_mean"] for r in p_rows], dtype=np.float64)
            ysd = np.asarray([r["ms_per_step_std"] for r in p_rows], dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=ysd,
                marker="o",
                linewidth=2.0,
                color="#4C78A8",
                capsize=3,
                label="Observed",
            )
            alpha = float(compute.get("particle_fit_alpha", 0.0))
            coeff = float(compute.get("particle_fit_coeff", 0.0))
            if coeff > 0.0:
                xfit = np.linspace(np.min(x), np.max(x), 120)
                yfit = coeff * np.power(xfit, alpha)
                ax.plot(xfit, yfit, "--", color="gray", label=f"fit alpha={alpha:.3f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_title("Step cost vs particles")
        ax.set_xlabel("Particles N")
        ax.set_ylabel("ms / step")
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        if g_rows:
            labels = [f"{int(r['grid_size'][0])}^3" for r in g_rows]
            mu = [float(r["ms_per_step_mean"]) for r in g_rows]
            sd = [float(r["ms_per_step_std"]) for r in g_rows]
            x = np.arange(len(labels), dtype=np.float64)
            ax.bar(x, mu, yerr=sd, color="#59A14F", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        ax.set_title("Step cost vs grid")
        ax.set_xlabel("Grid size")
        ax.set_ylabel("ms / step")
        ax.grid(True, axis="y", alpha=0.2)

        ax = axes[2]
        if l_rows:
            x = np.asarray([r["seq_len"] for r in l_rows], dtype=np.float64)
            y = np.asarray([r["ms_per_step_mean"] for r in l_rows], dtype=np.float64)
            ysd = np.asarray([r["ms_per_step_std"] for r in l_rows], dtype=np.float64)
            ax.errorbar(
                x, y, yerr=ysd, marker="o", linewidth=2.0, color="#B07AA1", capsize=3
            )
            ax.set_xscale("log", base=2)
            a = float(latency.get("fit_alpha", 0.0))
            lo = float(latency.get("fit_alpha_ci_low", 0.0))
            hi = float(latency.get("fit_alpha_ci_high", 0.0))
            ax.text(
                0.03,
                0.95,
                f"alpha={a:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        ax.set_title("Latency vs sequence length")
        ax.set_xlabel("Sequence length N")
        ax.set_ylabel("ms / step")
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / f"scaling_compute.{self.config.format}"
        fig.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return {"status": "success", "path": str(out)}
