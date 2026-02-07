"""Paper-ready artifacts for the collision experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
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
            # Folding evidence (map-vs-path metrics)
            # -------------------------------------------------------------
            fold_top1 = row.get("fold_top1")
            fold_pr = row.get("fold_pr")
            fold_entropy = row.get("fold_entropy")
            trans_top1 = row.get("transition_top1_prob")
            recall_top1 = row.get("recall_top1")
            recall_top3 = row.get("recall_top3")
            recall_mrr = row.get("recall_mrr")
            unique_keys = row.get("unique_keys")
            n_tokens = row.get("n_tokens")

            fold_vals = [
                float(fold_top1) if isinstance(fold_top1, (int, float)) else 0.0,
                float(trans_top1) if isinstance(trans_top1, (int, float)) else 0.0,
            ]
            pr_entropy_vals = [
                float(fold_pr) if isinstance(fold_pr, (int, float)) else 0.0,
                float(fold_entropy) if isinstance(fold_entropy, (int, float)) else 0.0,
            ]
            recall_vals = [
                float(recall_top1) if isinstance(recall_top1, (int, float)) else 0.0,
                float(recall_top3) if isinstance(recall_top3, (int, float)) else 0.0,
                float(recall_mrr) if isinstance(recall_mrr, (int, float)) else 0.0,
            ]
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.0, 3.6))
            ax0.bar(
                ["Fold Top-1", "Path Top-1"],
                fold_vals,
                color=["#5B8C5A", "#4C78A8"],
                alpha=0.9,
            )
            ax0.set_ylim(0.0, 1.0)
            ax0.set_ylabel("Probability")
            ax0.set_title("Collision folding concentration")
            ax0.grid(True, axis="y", alpha=0.25)

            ax1.bar(
                ["Fold PR", "Fold Entropy"],
                pr_entropy_vals,
                color=["#F28E2B", "#E15759"],
                alpha=0.9,
            )
            ax1.set_ylabel("Value")
            ax1.set_title("Folding spread vs entropy")
            ax1.grid(True, axis="y", alpha=0.25)

            ax2.bar(
                ["Recall@1", "Recall@3", "Recall MRR"],
                recall_vals,
                color=["#59A14F", "#76B7B2", "#EDC948"],
                alpha=0.9,
            )
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("Score")
            ax2.set_title("Trie recall test")
            ax2.grid(True, axis="y", alpha=0.25)

            summary_lines: list[str] = []
            if isinstance(n_tokens, (int, float)) and isinstance(
                unique_keys, (int, float)
            ):
                summary_lines.append(
                    f"Tokens: {int(n_tokens)}  Keys: {int(unique_keys)}"
                )
            if summary_lines:
                fig.text(0.5, -0.03, " | ".join(summary_lines), ha="center")

            out_base = f"{run}_folding_evidence"
            for fmt in self.config.formats:
                out_path = self.output_dir / f"{out_base}.{fmt}"
                fig.savefig(out_path, dpi=int(self.config.dpi), bbox_inches="tight")
                written.append(str(out_path))
            plt.close(fig)

            # -------------------------------------------------------------
            # Transition bifurcation sketch (top edges as path branches)
            # -------------------------------------------------------------
            top_edges = row.get("transitions_top_edges")
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            if isinstance(top_edges, Sequence) and len(top_edges) > 0:
                pos_levels: dict[int, dict[int, float]] = {}
                edge_rows: list[tuple[int, int, int, int, int]] = []
                max_count = 0
                for edge in top_edges:
                    if not isinstance(edge, dict):
                        continue
                    src = edge.get("src")
                    dst = edge.get("dst")
                    count = edge.get("count")
                    if not (
                        isinstance(src, dict)
                        and isinstance(dst, dict)
                        and isinstance(count, int)
                    ):
                        continue
                    src_pos = int(src.get("pos", 0))
                    dst_pos = int(dst.get("pos", 0))
                    src_byte = int(src.get("byte", 0))
                    dst_byte = int(dst.get("byte", 0))
                    pos_levels.setdefault(src_pos, {})
                    pos_levels.setdefault(dst_pos, {})
                    pos_levels[src_pos].setdefault(src_byte, 0.0)
                    pos_levels[dst_pos].setdefault(dst_byte, 0.0)
                    edge_rows.append((src_pos, src_byte, dst_pos, dst_byte, int(count)))
                    max_count = max(max_count, int(count))

                if edge_rows:
                    for pos, level in pos_levels.items():
                        keys = sorted(level.keys())
                        n = max(1, len(keys))
                        for i, key in enumerate(keys):
                            level[key] = float(i) - 0.5 * float(n - 1)

                    for src_pos, src_byte, dst_pos, dst_byte, count in edge_rows:
                        x0 = float(src_pos)
                        y0 = pos_levels[src_pos][src_byte]
                        x1 = float(dst_pos)
                        y1 = pos_levels[dst_pos][dst_byte]
                        width = 0.6 + (2.8 * float(count) / float(max(1, max_count)))
                        hue = (float(src_byte) % 256.0) / 255.0
                        color = plt.get_cmap("viridis")(hue)
                        ax.plot(
                            [x0, x1], [y0, y1], color=color, linewidth=width, alpha=0.55
                        )

                    for pos, level in pos_levels.items():
                        ys = [float(v) for v in level.values()]
                        ax.scatter(
                            [float(pos)] * len(ys),
                            ys,
                            s=26,
                            color="#2f2f2f",
                            alpha=0.9,
                            zorder=3,
                        )

                    min_pos = min(pos_levels.keys())
                    max_pos = max(pos_levels.keys())
                    ax.set_xlim(float(min_pos) - 0.5, float(max_pos) + 0.5)
                    y_abs = max(
                        abs(v) for level in pos_levels.values() for v in level.values()
                    )
                    y_lim = max(1.0, float(math.ceil(y_abs + 0.5)))
                    ax.set_ylim(-y_lim, y_lim)
                    ax.set_xlabel("Sequence position")
                    ax.set_ylabel("Branch lane")
                    ax.set_title("Top transition bifurcation sketch")
                    ax.grid(True, alpha=0.20)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No transition-edge data available for bifurcation sketch",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_title("Top transition bifurcation sketch")
                ax.set_axis_off()

            out_base = f"{run}_bifurcation"
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
