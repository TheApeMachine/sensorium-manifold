"""Projector that renders labelled ω-spectrum figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING

from sensorium.projectors.base import BaseProjector

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class OmegaLabelFigureConfig:
    name_prefix: str = "omega_labels"
    run_name_field: str = "run_name"
    omega_field: str = "spectrum_omega"
    amp_field: str = "spectrum_amp"
    labels_field: str = "omega_labels"
    sample_field: str = "omega_labels_sample_id"
    max_label_text: int = 24


class OmegaLabelFigureProjector(BaseProjector):
    """Write a spectrum plot with labelled ω markers per run."""

    def __init__(self, config: OmegaLabelFigureConfig | None = None, output_dir: Path | None = None):
        super().__init__(output_dir or Path("paper/figures"))
        self.config = config or OmegaLabelFigureConfig()

    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        import matplotlib.pyplot as plt
        import numpy as np

        self.ensure_output_dir()
        results = self._get_results_list(source)
        wrote: List[str] = []

        for r in results:
            run = r.get(self.config.run_name_field, "")
            omega = r.get(self.config.omega_field, None)
            amp = r.get(self.config.amp_field, None)
            labels = r.get(self.config.labels_field, None)
            sid = r.get(self.config.sample_field, None)
            if not isinstance(run, str) or not run:
                continue
            if not (isinstance(omega, list) and isinstance(amp, list) and len(omega) == len(amp) and len(omega) > 0):
                continue
            if not isinstance(labels, list) or not labels:
                continue

            x = np.asarray(omega, dtype=np.float64)
            y = np.asarray(amp, dtype=np.float64)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, y, color="#3498db", lw=1.5)
            sid_s = f" | sample={int(sid)}" if isinstance(sid, int) else ""
            ax.set_title(f"Labelled ω spectrum{sid_s} | {run}")
            ax.set_xlabel("ω")
            ax.set_ylabel("|Ψ(ω)|")
            ax.grid(True, alpha=0.25)

            # Add labelled markers.
            y_max = float(np.max(y)) if y.size else 1.0
            y_base = max(y_max * 0.05, 1e-8)

            for j, it in enumerate(labels):
                if not isinstance(it, dict):
                    continue
                om = it.get("omega", None)
                lab = it.get("label", "")
                if not isinstance(om, (int, float)):
                    continue
                if not isinstance(lab, str):
                    lab = str(lab)
                if len(lab) > int(self.config.max_label_text):
                    lab = lab[: int(self.config.max_label_text) - 1] + "…"

                ax.axvline(float(om), color="#f39c12", lw=1.0, alpha=0.7)
                ax.text(
                    float(om),
                    y_base * (1.2 + 0.15 * (j % 6)),
                    lab,
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#f39c12", ec="none", alpha=0.35),
                )

            out_path = self.output_dir / f"{self.config.name_prefix}_{run}.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            wrote.append(str(out_path))

        return {"status": "success", "wrote": wrote, "count": len(wrote)}

