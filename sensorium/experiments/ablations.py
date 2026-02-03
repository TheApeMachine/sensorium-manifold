"""
Ablation Study

Tests the contribution of individual components by disabling them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..core.config import PhysicsConfig
from ..semantic.hierarchical import HierarchicalSemanticManifold
from ..semantic.manifold import SemanticManifold


def run_single_condition(
    *,
    condition_name: str,
    steps: int,
    shift_at: int,
    context_len: int,
    dt: float,
    device: torch.device,
    use_hierarchy: bool = True,
    use_pondering: bool = True,
    use_homeostasis: bool = True,
) -> Dict[str, Any]:
    """Run a single ablation condition."""
    
    vocab = ["<bos>", "the", "cat", "sat", "on", "mat", "<eos>"]
    tid = {t: i for i, t in enumerate(vocab)}
    
    fwd = [tid["<bos>"], tid["the"], tid["cat"], tid["sat"], tid["on"], tid["the"], tid["mat"], tid["<eos>"]]
    rev = [tid["<bos>"], tid["mat"], tid["the"], tid["on"], tid["sat"], tid["cat"], tid["the"], tid["<eos>"]]
    
    # Pre-generate stream
    stream: List[int] = []
    pos = 0
    seq = fwd
    for t in range(steps + 1):
        if t == shift_at:
            seq = rev
            pos = 0
        stream.append(seq[pos])
        pos = (pos + 1) % len(seq)
    
    # Configure based on ablation
    cfg = PhysicsConfig(
        dt=dt,
        eps=1e-8,
        tau=1.0 if use_homeostasis else 1000.0,  # Very slow homeostasis = effectively off
    )
    
    if use_hierarchy:
        brain = HierarchicalSemanticManifold(
            cfg, device,
            vocab=vocab,
            embed_dim=min(16, len(vocab)),
            chunk_min_len=2,
            chunk_max_len=4,
        )
    else:
        brain = SemanticManifold(
            config=cfg,
            device=device,
            vocab=vocab,
            embed_dim=min(16, len(vocab)),
        )
    
    history: List[int] = []
    acc = torch.zeros(steps, dtype=torch.float32)
    
    for t in range(steps):
        cur = int(stream[t])
        nxt = int(stream[t + 1])
        
        history.append(cur)
        if len(history) > context_len:
            history = history[-context_len:]
        
        ctx = torch.tensor(history, device=device, dtype=torch.long)
        brain.ingest_ids(ctx)
        brain.step_grammar()
        out = brain.output_state()
        
        pred = int(out.token_index)
        acc[t] = 1.0 if pred == nxt else 0.0
        
        brain.observe_next_token(nxt, probs=out.probs)
        
        if use_pondering:
            brain.idle_think(steps=1, dream_steps=context_len)
    
    # Rolling accuracy
    win = max(1, len(fwd))
    kern = torch.ones(win, dtype=torch.float32) / float(win)
    acc_smooth = torch.nn.functional.conv1d(
        acc.view(1, 1, -1), kern.view(1, 1, -1), padding=win // 2
    ).view(-1)[:steps]
    
    pre_shift_acc = float(acc_smooth[shift_at - 100:shift_at].mean().item())
    post_shift_acc_recovered = float(acc_smooth[-100:].mean().item())
    
    # Find recovery
    threshold = pre_shift_acc * 0.8
    recovery_step = None
    for t in range(shift_at, min(shift_at + 500, steps)):
        if float(acc_smooth[t].item()) >= threshold:
            recovery_step = t - shift_at
            break
    
    return {
        "condition": condition_name,
        "pre_shift_accuracy": pre_shift_acc,
        "post_shift_accuracy": post_shift_acc_recovered,
        "recovery_steps": recovery_step,
    }


def generate_ablation_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table for ablation results."""
    rows = []
    for r in results:
        recovery = str(r['recovery_steps']) if r['recovery_steps'] else "$>$500"
        rows.append(
            f"    {r['condition']} & {r['pre_shift_accuracy']:.1%} & {r['post_shift_accuracy']:.1%} & {recovery} \\\\"
        )
    
    return r"""\begin{table}[t]
\centering
\caption{Ablation study. Each row disables one component from the full system.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Condition} & \textbf{Pre-shift Acc.} & \textbf{Post-shift Acc.} & \textbf{Recovery Steps} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def run_ablation_study(
    device: torch.device,
    tables_dir: Path,
    figures_dir: Path,
    steps: int = 2000,
    shift_at: int = 1000,
    context_len: int = 6,
    dt: float = 0.02,
):
    """Run all ablation conditions and generate table."""
    from .harness import ExperimentResult
    
    conditions = [
        ("Full system", True, True, True),
        ("No hierarchy", False, True, True),
        ("No pondering", True, False, True),
        ("No homeostasis", True, True, False),
    ]
    
    results = []
    for name, hier, pond, homeo in conditions:
        print(f"  Running: {name}...")
        result = run_single_condition(
            condition_name=name,
            steps=steps,
            shift_at=shift_at,
            context_len=context_len,
            dt=dt,
            device=device,
            use_hierarchy=hier,
            use_pondering=pond,
            use_homeostasis=homeo,
        )
        results.append(result)
        print(f"    Pre: {result['pre_shift_accuracy']:.1%}, "
              f"Post: {result['post_shift_accuracy']:.1%}, "
              f"Recovery: {result['recovery_steps']}")
    
    # Generate table
    table_content = generate_ablation_table(results)
    table_path = tables_dir / "ablation.tex"
    table_path.write_text(table_content)
    print(f"  [TABLE] ablation.tex")
    
    # Combine metrics
    metrics = {
        "conditions": results,
        "full_pre": results[0]["pre_shift_accuracy"],
        "full_post": results[0]["post_shift_accuracy"],
        "full_recovery": results[0]["recovery_steps"],
    }
    
    return ExperimentResult(
        name="Ablation Study",
        metrics=metrics,
        tables={"ablation": table_content},
        figures={},
    )
