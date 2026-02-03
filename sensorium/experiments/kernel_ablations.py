"""Kernel-based ablation study (Metal/MPS).

Writes:
- `paper/tables/ablation.tex`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from optimizer.manifold_physics import SpectralCarrierConfig

from .kernel_rule_shift import KernelRuleShiftConfig, run_kernel_rule_shift


def _row(name: str, pre: float, post: float, rec: Optional[int]) -> str:
    rs = str(rec) if rec is not None else "N/A"
    return f"{name} & {pre:.3f} & {post:.3f} & {rs} \\\\"


def run_kernel_ablation_study(
    *,
    device: str = "mps",
    out_dir: Path = Path("./paper"),
    base_cfg: Optional[KernelRuleShiftConfig] = None,
) -> Dict[str, Any]:
    cfg = base_cfg or KernelRuleShiftConfig(device=device)

    conditions: List[Tuple[str, SpectralCarrierConfig]] = []

    # Full system (defaults)
    conditions.append(("Full", SpectralCarrierConfig()))

    # No top-down bias
    conditions.append(
        (
            "No top-down",
            SpectralCarrierConfig(
                topdown_phase_scale=0.0,
                topdown_energy_scale=0.0,
                topdown_random_energy_eps=0.0,
            ),
        )
    )

    # No crystallization (effectively disables long-term memory)
    conditions.append(
        (
            "No crystallization",
            SpectralCarrierConfig(
                stable_amp_threshold=1e9,
                crystallize_amp_threshold=1e9,
                crystallize_age=10**9,
                crystallized_coupling_boost=0.0,
            ),
        )
    )

    # No splitting (forces single-spectrum compromise)
    conditions.append(
        (
            "No splitting",
            SpectralCarrierConfig(
                conflict_threshold=1e9,
            ),
        )
    )

    # No exploration (anchors become greedy)
    conditions.append(
        (
            "No exploration",
            SpectralCarrierConfig(
                anchor_random_eps=0.0,
                topdown_random_energy_eps=0.0,
            ),
        )
    )

    rows: List[str] = []
    results: List[Dict[str, Any]] = []
    for name, scfg in conditions:
        m = run_kernel_rule_shift(cfg, spectral_cfg=scfg, out_dir=out_dir)
        pre = float(m["pre_shift_alignment"])
        post = float(m["post_shift_alignment_immediate"])
        rec = m["recovery_steps"]
        rows.append(_row(name, pre, post, rec))
        results.append({"condition": name, "pre": pre, "post": post, "recovery_steps": rec})

    table = r"""\begin{table}[t]
\centering
\caption{Kernel ablations. We disable individual carrier-memory mechanisms and report alignment before/after rule reversal.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Condition} & \textbf{Pre-shift} & \textbf{Post-shift} & \textbf{Recovery steps} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "ablation.tex").write_text(table, encoding="utf-8")

    return {"conditions": results}

