from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Tuple
import json
import math
import os

import torch


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False
    ):
        self.experiment_name = experiment_name
        self.profile = profile

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Stable artifact layout (anchored to repo root, not cwd).
        self.repo_root = Path(__file__).resolve().parents[2]
        self.paper_dir = self.repo_root / "paper"
        self.artifacts_dir = self.paper_dir / "artifacts"
        self.tables_dir = self.paper_dir / "tables"
        self.figures_dir = self.paper_dir / "figures"
        self.video_dir = self.artifacts_dir / "video"

        self.video_path = self.video_dir / f"{experiment_name}_{self.timestamp}.mp4"
        # Default location for any manifold dashboards/videos during this experiment.
        # This allows `ManifoldConfig(dashboard=True)` to record without extra boilerplate.
        os.environ["THERMO_MANIFOLD_EXPERIMENT_NAME"] = str(experiment_name)
        os.environ["THERMO_MANIFOLD_DASHBOARD_VIDEO_PATH"] = str(self.video_path)

    @abstractmethod
    def observe(self, state: dict):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement this method")

    # ---------------------------------------------------------------------
    # Artifact helpers
    # ---------------------------------------------------------------------
    def artifact_path(self, kind: str, filename: str) -> Path:
        """Return an on-disk artifact path, creating parent dirs on write.

        `kind` is typically one of: "tables", "figures", "artifacts", "video".
        """
        if kind == "tables":
            return self.tables_dir / filename
        if kind == "figures":
            return self.figures_dir / filename
        if kind == "video":
            return self.video_dir / filename
        if kind == "artifacts":
            return self.artifacts_dir / filename
        # Allow custom subfolders under paper/ by convention.
        return self.paper_dir / kind / filename

    def write_text(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, path: Path, obj: Any, *, indent: int = 2) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    # ---------------------------------------------------------------------
    # LaTeX tables (paper-ready)
    # Convention: experiments produce `paper/tables/<name>.tex`.
    # ---------------------------------------------------------------------
    @staticmethod
    def latex_escape(s: Any) -> str:
        """Escape common LaTeX special chars for table content."""
        text = "" if s is None else str(s)
        # Minimal escape set for tables.
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}")
        )

    @staticmethod
    def _format_number(x: Any, *, floatfmt: str = ".4g") -> str:
        if x is None:
            return ""
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, (int,)):
            return str(x)
        if isinstance(x, (float,)):
            if math.isnan(x):
                return "NaN"
            if math.isinf(x):
                return "\\infty" if x > 0 else "-\\infty"
            return format(x, floatfmt)
        return str(x)

    def latex_kv_table(
        self,
        metrics: Mapping[str, Any],
        *,
        key_header: str = "Metric",
        value_header: str = "Value",
        floatfmt: str = ".4g",
        sort_keys: bool = True,
    ) -> str:
        """Two-column key/value table. Ideal for experiment summaries."""
        items = list(metrics.items())
        if sort_keys:
            items.sort(key=lambda kv: kv[0])

        lines: list[str] = []
        lines.append("\\begin{tabular}{l r}")
        lines.append("\\toprule")
        lines.append(f"{self.latex_escape(key_header)} & {self.latex_escape(value_header)} \\\\")
        lines.append("\\midrule")
        for k, v in items:
            kk = self.latex_escape(k)
            vv = self.latex_escape(self._format_number(v, floatfmt=floatfmt))
            lines.append(f"{kk} & {vv} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines) + "\n"

    def latex_column_table(
        self,
        columns: Mapping[str, Sequence[Any]],
        *,
        floatfmt: str = ".4g",
        align: str | None = None,
        row_labels: Sequence[str] | None = None,
    ) -> str:
        """Columnar table from dict-of-columns.

        Convention: `columns` maps column name -> sequence of per-row values.
        All columns must have the same length. If provided, `row_labels` adds a
        leading label column.
        """
        col_names = list(columns.keys())
        if not col_names:
            raise ValueError("columns must not be empty")

        n_rows = len(columns[col_names[0]])
        for name in col_names[1:]:
            if len(columns[name]) != n_rows:
                raise ValueError("all columns must have the same length")
        if row_labels is not None and len(row_labels) != n_rows:
            raise ValueError("row_labels length must match number of rows")

        n_data_cols = len(col_names) + (1 if row_labels is not None else 0)
        tab_align = align or ("l" + "r" * (n_data_cols - 1))

        lines: list[str] = []
        lines.append(f"\\begin{{tabular}}{{{tab_align}}}")
        lines.append("\\toprule")

        header_cells: list[str] = []
        if row_labels is not None:
            header_cells.append("")  # empty corner cell
        header_cells.extend(self.latex_escape(n) for n in col_names)
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for i in range(n_rows):
            row: list[str] = []
            if row_labels is not None:
                row.append(self.latex_escape(row_labels[i]))
            for name in col_names:
                row.append(self.latex_escape(self._format_number(columns[name][i], floatfmt=floatfmt)))
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines) + "\n"

    def write_table_tex(self, name: str, table_tex: str) -> Path:
        """Write `paper/tables/<name>.tex`."""
        if not name.endswith(".tex"):
            name = f"{name}.tex"
        return self.write_text(self.artifact_path("tables", name), table_tex)

    def write_kv_table(
        self,
        name: str,
        metrics: Mapping[str, Any],
        *,
        key_header: str = "Metric",
        value_header: str = "Value",
        floatfmt: str = ".4g",
        sort_keys: bool = True,
    ) -> Path:
        """Convenience: render + write a key/value summary table."""
        return self.write_table_tex(
            name,
            self.latex_kv_table(
                metrics,
                key_header=key_header,
                value_header=value_header,
                floatfmt=floatfmt,
                sort_keys=sort_keys,
            ),
        )

    def write_columnar_table(
        self,
        name: str,
        columns: Mapping[str, Sequence[Any]],
        *,
        floatfmt: str = ".4g",
        align: str | None = None,
        row_labels: Sequence[str] | None = None,
    ) -> Path:
        """Convenience: render + write a dict-of-columns table."""
        return self.write_table_tex(
            name,
            self.latex_column_table(
                columns,
                floatfmt=floatfmt,
                align=align,
                row_labels=row_labels,
            ),
        )

    # ---------------------------------------------------------------------
    # Standard visualizations (paper-ready)
    # Convention: experiments produce `paper/figures/<name>.pdf|png`.
    # These helpers intentionally import matplotlib lazily.
    # ---------------------------------------------------------------------
    def plot_series(
        self,
        series: Mapping[str, Sequence[float]],
        *,
        name: str,
        x: Sequence[float] | None = None,
        title: str | None = None,
        xlabel: str = "step",
        ylabel: str = "value",
        logy: bool = False,
        figsize: tuple[float, float] = (6.0, 3.5),
        fmt: str = "pdf",
    ) -> Path:
        import matplotlib.pyplot as plt  # type: ignore

        path = self.artifact_path("figures", f"{name}.{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=figsize)
        for label, y in series.items():
            xx = range(len(y)) if x is None else x
            plt.plot(list(xx), list(y), label=label)
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if logy:
            plt.yscale("log")
        if len(series) > 1:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def plot_bar(
        self,
        values: Mapping[str, float],
        *,
        name: str,
        title: str | None = None,
        xlabel: str = "",
        ylabel: str = "value",
        figsize: tuple[float, float] = (6.0, 3.5),
        fmt: str = "pdf",
        sort: bool = True,
    ) -> Path:
        import matplotlib.pyplot as plt  # type: ignore

        items = list(values.items())
        if sort:
            items.sort(key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in items]
        ys = [float(v) for _, v in items]

        path = self.artifact_path("figures", f"{name}.{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=figsize)
        plt.bar(range(len(ys)), ys)
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def plot_heatmap(
        self,
        matrix: Sequence[Sequence[float]],
        *,
        name: str,
        title: str | None = None,
        xlabel: str = "",
        ylabel: str = "",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (5.5, 4.0),
        fmt: str = "pdf",
        show_values: bool = False,
        value_fmt: str = ".2g",
    ) -> Path:
        import matplotlib.pyplot as plt  # type: ignore

        path = self.artifact_path("figures", f"{name}.{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=figsize)
        im = plt.imshow(matrix, aspect="auto", cmap=cmap)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if show_values:
            for i, row in enumerate(matrix):
                for j, v in enumerate(row):
                    plt.text(j, i, format(float(v), value_fmt), ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path