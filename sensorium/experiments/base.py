from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence

from sensorium.projectors import (
    PipelineProjector,
    LaTeXTableProjector,
    ConsoleProjector,
    TopTransitionsProjector,
    TableConfig,
)


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
        reportable: Sequence[str] | None = None,
    ):
        self.experiment_name = slugify(experiment_name)
        self.profile = profile
        self.dashboard = dashboard
        self.reportable = list(reportable) if reportable else []
        self.repo_root = Path(__file__).resolve().parents[2]
        self._dashboard_instance = None
        self._artifact_dir = self.repo_root / "artifacts" / self.experiment_name
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        # Default: keep videos next to paper assets.
        self.video_path = str(
            self.repo_root
            / "paper"
            / "videos"
            / f"{self.experiment_name}_dashboard.mp4"
        )

        self.projector = PipelineProjector(
            # Console output for real-time feedback
            ConsoleProjector(
                fields=self.reportable,
                format="table",
            ),
            TopTransitionsProjector(),
            # LaTeX table - columns match InferenceObserver field names
            LaTeXTableProjector(
                TableConfig(
                    name=f"{self.experiment_name}_summary",
                    columns=self.reportable,
                    caption=f"{experiment_name} metrics",
                    label=f"tab:{self.experiment_name}",
                    precision=3,
                ),
                output_dir=Path("paper/tables"),
            ),
        )

    def _slug(self) -> str:
        return "".join(
            ch if ch.isalnum() else "_" for ch in self.experiment_name.lower()
        ).strip("_")

    def experiment_seeds(self, default: Sequence[int] = (42, 43, 44, 45, 46)) -> list[int]:
        """Return deterministic experiment seeds from env or defaults.

        Env override:
          SENSORIUM_EXPERIMENT_SEEDS="42,43,44"
        """
        raw = str(os.getenv("SENSORIUM_EXPERIMENT_SEEDS", "")).strip()
        if not raw:
            return [int(x) for x in default]

        out: list[int] = []
        seen: set[int] = set()
        for part in raw.replace(";", ",").replace(" ", ",").split(","):
            token = part.strip()
            if not token:
                continue
            value = int(token)
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        if not out:
            raise ValueError(
                "SENSORIUM_EXPERIMENT_SEEDS was set but no valid seeds were parsed."
            )
        return out

    def assert_allowed_backends(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        allowed: Sequence[str] = ("mps",),
        backend_field: str = "run_backend",
    ) -> None:
        """Fail hard when publication rows include disallowed backends."""
        allowed_set = {str(x) for x in allowed}
        bad = [
            r
            for r in rows
            if str(r.get(backend_field, "")).strip() not in allowed_set
        ]
        if not bad:
            return
        samples = []
        for row in bad[:8]:
            samples.append(
                {
                    "scenario": row.get("scenario", ""),
                    "run_name": row.get("run_name", ""),
                    backend_field: row.get(backend_field, ""),
                }
            )
        raise RuntimeError(
            "Publication artifacts require real manifold backend(s) "
            f"{sorted(allowed_set)}. Found disallowed rows: {samples}"
        )

    def write_provenance_jsonl(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        stem: str | None = None,
    ) -> Path:
        """Write per-run scalar provenance rows for auditing/reproducibility."""
        out_path = (
            self.repo_root
            / "paper"
            / "artifacts"
            / f"{(stem or self.experiment_name)}_provenance.jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        git_commit = self._git_commit()
        ts = datetime.now(timezone.utc).isoformat()

        lines: list[str] = []
        for row in rows:
            record: dict[str, Any] = {
                "timestamp_utc": ts,
                "experiment_name": self.experiment_name,
                "git_commit": git_commit,
            }
            for key, value in row.items():
                if isinstance(value, bool):
                    record[str(key)] = bool(value)
                elif isinstance(value, (int, float, str)) or value is None:
                    record[str(key)] = value
            lines.append(json.dumps(record, sort_keys=True))

        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return out_path

    def aggregate_rows_with_ci(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        group_field: str = "scenario",
        metric_fields: Sequence[str],
        carry_fields: Sequence[str] = (),
        seed_field: str = "seed",
    ) -> list[dict[str, Any]]:
        """Aggregate metric means and 95% CI per scenario/group."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(group_field, ""))
            groups.setdefault(key, []).append(row)

        out: list[dict[str, Any]] = []
        for key in sorted(groups.keys()):
            bucket = groups[key]
            item: dict[str, Any] = {group_field: key}

            seed_values = [
                int(r[seed_field])
                for r in bucket
                if isinstance(r.get(seed_field), (int, float))
            ]
            if seed_values:
                item["n_seeds"] = int(len(set(seed_values)))
            else:
                item["n_seeds"] = int(len(bucket))

            for field in carry_fields:
                values = [r.get(field) for r in bucket if r.get(field) is not None]
                if not values:
                    continue
                first = values[0]
                if all(v == first for v in values):
                    item[field] = first
                else:
                    item[field] = "mixed"

            for metric in metric_fields:
                values = [
                    float(r[metric])
                    for r in bucket
                    if isinstance(r.get(metric), (int, float))
                    and not isinstance(r.get(metric), bool)
                    and math.isfinite(float(r[metric]))
                ]
                if not values:
                    continue
                n = float(len(values))
                mean = float(sum(values) / n)
                if len(values) > 1:
                    var = float(
                        sum((v - mean) * (v - mean) for v in values)
                        / float(len(values) - 1)
                    )
                    std = math.sqrt(max(var, 0.0))
                    ci95 = float(1.96 * std / math.sqrt(n))
                else:
                    ci95 = 0.0
                item[f"{metric}_mean"] = mean
                item[f"{metric}_ci95"] = ci95

            out.append(item)
        return out

    def ci_summary_columns(
        self, metric_fields: Sequence[str], *, prefix_fields: Sequence[str] = ("scenario", "n_seeds")
    ) -> list[str]:
        cols = [str(x) for x in prefix_fields]
        for metric in metric_fields:
            cols.append(f"{metric}_mean")
            cols.append(f"{metric}_ci95")
        return cols

    def infer_numeric_fields(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        candidate_fields: Sequence[str],
        exclude_fields: Sequence[str] = (),
    ) -> list[str]:
        out: list[str] = []
        excluded = set(exclude_fields)
        for field in candidate_fields:
            if field in excluded:
                continue
            seen_numeric = any(
                isinstance(r.get(field), (int, float))
                and not isinstance(r.get(field), bool)
                for r in rows
            )
            if seen_numeric:
                out.append(field)
        return out

    def _git_commit(self) -> str:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except Exception:
            return "unknown"

    def artifact_path(self, *parts: str) -> Path:
        """Resolve an artifact path and create parent directories."""
        if parts and parts[0] in {"tables", "figures"}:
            base = self.repo_root / "paper"
        else:
            base = self._artifact_dir
        path = base.joinpath(*parts) if parts else base
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def close_dashboard(self):
        """Stop recording and close the dashboard."""
        if self._dashboard_instance is not None:
            self._dashboard_instance.stop_recording()
            self._dashboard_instance.close()
            self._dashboard_instance = None
            print(f"[dashboard] Video saved to {self.video_path}")

    def start_dashboard(
        self, *, grid_size: tuple[int, int, int], run_name: Optional[str] = None
    ) -> None:
        """Start live dashboard + mp4 recording for this run."""
        from sensorium.instrument.dashboard import DashboardSession

        # Restart if already running.
        self.close_dashboard()

        name = (run_name or self.experiment_name).strip() or self.experiment_name
        video_path = (
            self.repo_root / "paper" / "videos" / f"{slugify(name)}_dashboard.mp4"
        )
        self.video_path = str(video_path)
        gx, gy, gz = grid_size
        self._dashboard_instance = DashboardSession.from_env(
            grid_size=(int(gx), int(gy), int(gz)),
            video_path=video_path,
        )

    def dashboard_update(self, state: dict) -> None:
        if self._dashboard_instance is None:
            return
        self._dashboard_instance.update(state)

    @abstractmethod
    def observe(self, state: dict):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
