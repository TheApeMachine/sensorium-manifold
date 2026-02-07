"""Rule-shift projectors for tables and figures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np

from sensorium.projectors.base import BaseProjector


@dataclass
class RuleShiftFigureConfig:
    """Configuration for rule-shift figures."""

    name: str = "rule_shift"
    format: str = "png"
    dpi: int = 300


class RuleShiftTableProjector(BaseProjector):
    """Project scaled rule-shift summary table with transparent conditions."""

    def __init__(
        self, output_dir: Path | None = None, name: str = "rule_shift_summary"
    ):
        super().__init__(output_dir or Path("paper/tables"))
        self.name = name

    @staticmethod
    def _fmt_pct(value: float) -> str:
        return f"{100.0 * float(value):.1f}\\%"

    @staticmethod
    def _mean_std(values: Sequence[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    @staticmethod
    def _ci95(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        if n <= 1:
            return 0.0
        return float(1.96 * np.std(arr, ddof=1) / np.sqrt(float(n)))

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        results = self._get_results_list(source)
        if not results:
            return {"status": "skipped", "reason": "no results"}

        self.ensure_output_dir()
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        backend_counts: dict[str, int] = defaultdict(int)
        for row in results:
            by_scenario[str(row.get("scenario", "unknown"))].append(row)
            backend_counts[str(row.get("run_backend", "unknown"))] += 1

        provenance_path = str(results[0].get("provenance_jsonl", ""))
        provenance_tex = (
            provenance_path.replace("\\", "/")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
        )

        header = r"""\begin{table*}[t]
\centering
\caption{Scaled rule-shift matrix with transparent conditions and adaptation metrics (mean $\pm$ std over seeds; Top-1 CI95 reported).}
\label{tab:rule_shift}
\begin{tabular}{l l c c c c c c c c c c}
\toprule
\textbf{Scenario} & \textbf{Domain} & \textbf{Seeds} & \textbf{Phases} & \textbf{Reps} & \textbf{Seg} & \textbf{Drop} & \textbf{Recovery} & \textbf{Final@1} & \textbf{Top1 CI95} & \textbf{Final@3} & \textbf{Final MRR} \\
\midrule
"""
        lines: list[str] = []
        for scenario_name in sorted(by_scenario.keys()):
            group = by_scenario[scenario_name]
            first = group[0]
            domain = str(first.get("domain", "n/a"))
            n_seeds = len({int(r.get("seed", 0)) for r in group})
            n_phases = int(first.get("n_phases", 0))
            total_reps = int(first.get("total_reps", 0))
            segment_size = int(first.get("segment_size", 0))

            drops = [abs(float(r.get("worst_drop_top1", 0.0))) for r in group]
            recoveries = [
                float(r.get("mean_recovery_reps", -1.0))
                for r in group
                if float(r.get("mean_recovery_reps", -1.0)) >= 0.0
            ]
            final_top1 = [float(r.get("final_top1", 0.0)) for r in group]
            final_top3 = [float(r.get("final_top3", 0.0)) for r in group]
            final_mrr = [float(r.get("final_mrr", 0.0)) for r in group]

            drop_mu, drop_sd = self._mean_std(drops)
            rec_mu, rec_sd = self._mean_std(recoveries)
            t1_mu, t1_sd = self._mean_std(final_top1)
            t3_mu, t3_sd = self._mean_std(final_top3)
            mrr_mu, mrr_sd = self._mean_std(final_mrr)
            t1_ci95 = self._ci95(final_top1)
            rec_txt = f"{rec_mu:.1f} $\\pm$ {rec_sd:.1f}" if recoveries else "N/A"

            lines.append(
                f"{scenario_name.replace('_', r'\_')} & "
                f"{domain.replace('_', r'\_')} & "
                f"{n_seeds} & {n_phases} & {total_reps} & {segment_size} & "
                f"{self._fmt_pct(drop_mu)} $\\pm$ {self._fmt_pct(drop_sd)} & "
                f"{rec_txt} & "
                f"{self._fmt_pct(t1_mu)} $\\pm$ {self._fmt_pct(t1_sd)} & "
                f"{self._fmt_pct(t1_ci95)} & "
                f"{self._fmt_pct(t3_mu)} $\\pm$ {self._fmt_pct(t3_sd)} & "
                f"{self._fmt_pct(mrr_mu)} $\\pm$ {self._fmt_pct(mrr_sd)} \\\\"
            )

        backend_text = ", ".join(
            f"{k}:{int(v)}" for k, v in sorted(backend_counts.items())
        )
        footer = r"""\bottomrule
\end{tabular}
\vspace{2pt}

""" + (
            r"{\footnotesize Conditions are explicit: number of phases, total repetitions, segment length, and seed count are reported per scenario. "
            + rf"Backend counts: {backend_text}. "
            + rf"Provenance log: \texttt{{{provenance_tex}}}.}}"
            + "\n"
        ) + r"""
\end{table*}
"""
        output_path = self.output_dir / f"{self.name}.tex"
        output_path.write_text(header + "\n".join(lines) + "\n" + footer)
        return {"status": "success", "path": str(output_path)}


class RuleShiftFigureProjector(BaseProjector):
    """Rule-shift matrix figure projector with real-world focus panel."""

    def __init__(
        self,
        config: RuleShiftFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(output_dir or Path("paper/figures"))
        if config:
            self.config = config
        else:
            self.config = RuleShiftFigureConfig(**kwargs)

    @staticmethod
    def _scenario_groups(
        results: Sequence[Dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            groups[str(row.get("scenario", "unknown"))].append(row)
        return groups

    @staticmethod
    def _history_mean(
        group: Sequence[Dict[str, Any]], key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        histories = [
            list(row.get("accuracy_history", []))
            for row in group
            if row.get("accuracy_history")
        ]
        if not histories:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        n = min(len(hist) for hist in histories)
        if n <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        reps = np.array(
            [float(histories[0][i].get("rep", 0.0)) for i in range(n)], dtype=np.float64
        )
        mat = np.array(
            [
                [float(history[i].get(key, 0.0)) for i in range(n)]
                for history in histories
            ],
            dtype=np.float64,
        )
        return reps, np.mean(mat, axis=0)

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = self._get_results_list(source)
        if not results:
            return {"status": "skipped", "reason": "no results"}

        groups = self._scenario_groups(results)
        if not groups:
            return {"status": "skipped", "reason": "no grouped results"}

        self.ensure_output_dir()
        scenario_names = sorted(groups.keys())
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2))

        ax = axes[0, 0]
        for name in scenario_names:
            reps, top1 = self._history_mean(groups[name], "top1")
            if reps.size:
                ax.plot(reps, top1, linewidth=2.0, label=name.replace("_", " "))
        ax.set_title("Adaptation trajectories (Top-1)")
        ax.set_xlabel("Repetition")
        ax.set_ylabel("Top-1 accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")

        ax = axes[0, 1]
        x = np.arange(len(scenario_names), dtype=np.float64)
        t1 = [
            np.mean([float(r.get("final_top1", 0.0)) for r in groups[name]])
            for name in scenario_names
        ]
        t3 = [
            np.mean([float(r.get("final_top3", 0.0)) for r in groups[name]])
            for name in scenario_names
        ]
        mrr = [
            np.mean([float(r.get("final_mrr", 0.0)) for r in groups[name]])
            for name in scenario_names
        ]
        w = 0.25
        ax.bar(x - w, t1, width=w, label="Final@1", color="#4C78A8")
        ax.bar(x, t3, width=w, label="Final@3", color="#59A14F")
        ax.bar(x + w, mrr, width=w, label="Final MRR", color="#F28E2B")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [name.replace("_", "\n") for name in scenario_names], fontsize=8
        )
        ax.set_ylim(0.0, 1.02)
        ax.set_title("Final recall quality")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        drops = [
            np.mean([abs(float(r.get("worst_drop_top1", 0.0))) for r in groups[name]])
            for name in scenario_names
        ]
        recov = []
        for name in scenario_names:
            vals = [float(r.get("mean_recovery_reps", -1.0)) for r in groups[name]]
            vals = [v for v in vals if v >= 0.0]
            recov.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(x, drops, "o-", linewidth=2, color="#E15759", label="Worst drop")
        ax.plot(x, recov, "s-", linewidth=2, color="#76B7B2", label="Recovery reps")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [name.replace("_", "\n") for name in scenario_names], fontsize=8
        )
        ax.set_title("Shift shock vs recovery speed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        strongest = max(
            scenario_names,
            key=lambda n: max(float(r.get("total_bytes", 0.0)) for r in groups[n]),
        )
        reps, top1 = self._history_mean(groups[strongest], "top1")
        _, top3 = self._history_mean(groups[strongest], "top3")
        _, mrr_curve = self._history_mean(groups[strongest], "mrr")
        ax.plot(reps, top1, linewidth=2.2, color="#4C78A8", label="Top-1")
        ax.plot(reps, top3, linewidth=2.2, color="#59A14F", label="Top-3")
        ax.plot(reps, mrr_curve, linewidth=2.2, color="#F28E2B", label="MRR")
        ax.set_title(f"Strongest evidence: {strongest.replace('_', ' ')}")
        ax.set_xlabel("Repetition")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        matrix_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        fig.savefig(matrix_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8.2, 4.1))
        ax2.plot(reps, top1, linewidth=2.4, color="#4C78A8", label="Top-1")
        ax2.plot(reps, top3, linewidth=2.4, color="#59A14F", label="Top-3")
        ax2.plot(reps, mrr_curve, linewidth=2.4, color="#F28E2B", label="MRR")
        ax2.set_title(f"Rule-shift recall detail: {strongest.replace('_', ' ')}")
        ax2.set_xlabel("Repetition")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0.0, 1.02)
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="lower right")
        plt.tight_layout()
        focus_path = (
            self.output_dir / f"{self.config.name}_realworld.{self.config.format}"
        )
        fig2.savefig(focus_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig2)

        return {
            "status": "success",
            "path": str(matrix_path),
            "focus_path": str(focus_path),
            "strongest_scenario": strongest,
        }
