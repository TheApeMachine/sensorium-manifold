"""Projector that writes top transition edges to JSON artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING

from sensorium.projectors.base import BaseProjector

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class TopTransitionsConfig:
    """Configure which fields to use and where to write."""

    run_name_field: str = "run_name"
    edges_field: str = "transitions_top_edges"
    output_subdir: str = "paper/artifacts"
    suffix: str = "_transitions_top.json"


class TopTransitionsProjector(BaseProjector):
    """Write per-run top transition edges JSON files.

    Expects each result row to contain:
    - run_name (string)
    - transitions_top_edges (small list[dict])
    """

    def __init__(self, config: TopTransitionsConfig | None = None):
        super().__init__(output_dir=None)
        self.config = config or TopTransitionsConfig()
        # Anchor to repo root (stable regardless of cwd).
        self.repo_root = Path(__file__).resolve().parents[2]

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        results = self._get_results_list(source)
        wrote: List[str] = []

        out_dir = (self.repo_root / self.config.output_subdir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            run = r.get(self.config.run_name_field, "")
            edges = r.get(self.config.edges_field, None)
            if not isinstance(run, str) or not run:
                continue
            if not isinstance(edges, list) or not edges:
                continue
            path = out_dir / f"{run}{self.config.suffix}"
            # Keep JSON small/stable: just top edges + total if present.
            blob = {
                "top_edges": edges,
                "total_edges": r.get("transition_edges", None),
            }
            path.write_text(__import__("json").dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            wrote.append(str(path))

        return {"status": "success", "wrote": wrote, "count": len(wrote)}

