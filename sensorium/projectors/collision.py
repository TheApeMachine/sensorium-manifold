"""Paper-ready artifacts for the collision experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from sensorium.projectors.base import BaseProjector


@dataclass(frozen=True, slots=True)
class CollisionFigureConfig:
    name_prefix: str = "collision"
    formats: tuple[str, ...] = ("pdf",)
    dpi: int = 200


class CollisionFigureProjector(BaseProjector):
    """Generate collision-specific figures from the latest observation."""

    def __init__(
        self,
        config: CollisionFigureConfig | None = None,
        *,
        output_dir: Path | None = None,
    ):
        super().__init__(output_dir=output_dir)
        self.config = config or CollisionFigureConfig()

    def project(self, source: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
        # Import matplotlib lazily (headless).
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.ensure_output_dir()

        if hasattr(source, "results"):
            rows = list(getattr(source, "results"))  # type: ignore[arg-type]
        elif isinstance(source, dict):
            rows = [source]
        else:
            rows = []

        written: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            run = row.get("run_name")
            if not isinstance(run, str) or not run:
                run = self.config.name_prefix

            # -------------------------------------------------------------
            # Collision multiplicity distribution
            # -------------------------------------------------------------
            xs = row.get("collision_mult_x")
            ys = row.get("collision_mult_y")
            if (
                isinstance(xs, Sequence)
                and isinstance(ys, Sequence)
                and len(xs) == len(ys)
                and len(xs) > 0
            ):
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                ax.bar(
                    [int(x) for x in xs],
                    [int(y) for y in ys],
                    width=0.85,
                    color="#2f6bff",
                    alpha=0.85,
                )
                ax.set_xlabel("Multiplicity (particles per token)")
                ax.set_ylabel("# Tokens")
                ax.set_title("Token collision multiplicity")
                ax.grid(True, axis="y", alpha=0.25)

                out_base = f"{run}_multiplicity"
                for fmt in self.config.formats:
                    out_path = self.output_dir / f"{out_base}.{fmt}"
                    fig.savefig(out_path, dpi=int(self.config.dpi), bbox_inches="tight")
                    written.append(str(out_path))
                plt.close(fig)

            # -------------------------------------------------------------
            # Wave spectrum snapshot
            # -------------------------------------------------------------
            omega = row.get("wave_omega")
            amp = row.get("wave_psi_amp")
            if (
                isinstance(omega, Sequence)
                and isinstance(amp, Sequence)
                and len(omega) == len(amp)
                and len(omega) > 0
            ):
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                ax.plot(
                    [float(x) for x in omega],
                    [float(y) for y in amp],
                    color="#ff7a1a",
                    linewidth=1.6,
                )
                ax.set_xlabel(r"$\omega$")
                ax.set_ylabel(r"$|\Psi(\omega)|$")
                ax.set_title("Coherence spectrum")
                ax.grid(True, alpha=0.25)

                out_base = f"{run}_spectrum"
                for fmt in self.config.formats:
                    out_path = self.output_dir / f"{out_base}.{fmt}"
                    fig.savefig(out_path, dpi=int(self.config.dpi), bbox_inches="tight")
                    written.append(str(out_path))
                plt.close(fig)

        return {"status": "success", "written": written, "count": len(written)}
