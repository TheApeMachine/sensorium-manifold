"""
Rule Shift Experiment

Tests the system's ability to adapt when the underlying sequential pattern
reverses mid-stream.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ..core.config import PhysicsConfig
from ..semantic.hierarchical import HierarchicalSemanticManifold


def _topological_entropy(src: torch.Tensor, w: torch.Tensor, eps: float) -> float:
    """Mean per-src entropy of outgoing normalized weights."""
    if src.numel() == 0:
        return 0.0
    w = w.clamp(min=0.0)
    if float(w.sum().item()) <= eps:
        return 0.0
    src_u, inv = torch.unique(src, return_inverse=True)
    out_sum = torch.zeros(int(src_u.numel()), device=w.device, dtype=w.dtype)
    out_sum.index_add_(0, inv, w)
    p = w / (out_sum[inv] + eps)
    edge_ent = -p * torch.log(p + eps)
    ent_src = torch.zeros(int(src_u.numel()), device=w.device, dtype=w.dtype)
    ent_src.index_add_(0, inv, edge_ent)
    return float(ent_src.mean().item())


def _make_stream(vocab: List[str]) -> Tuple[Dict[str, int], List[int], List[int]]:
    """Create forward and reverse token streams."""
    tid = {t: i for i, t in enumerate(vocab)}
    fwd = [
        tid["<bos>"], tid["the"], tid["cat"], tid["sat"],
        tid["on"], tid["the"], tid["mat"], tid["<eos>"],
    ]
    rev = [
        tid["<bos>"], tid["mat"], tid["the"], tid["on"],
        tid["sat"], tid["cat"], tid["the"], tid["<eos>"],
    ]
    return tid, fwd, rev


def run_rule_shift(
    *,
    steps: int,
    shift_at: int,
    context_len: int,
    dt: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run the rule shift experiment.
    
    Returns:
        Dictionary with all metrics and raw data
    """
    vocab = ["<bos>", "the", "cat", "sat", "on", "mat", "<eos>"]
    tid, fwd, rev = _make_stream(vocab)
    
    # Pre-generate the full token stream
    stream: List[int] = []
    pos = 0
    seq = fwd
    for t in range(steps + 1):
        if t == shift_at:
            seq = rev
            pos = 0
        stream.append(seq[pos])
        pos = (pos + 1) % len(seq)
    
    cfg = PhysicsConfig(dt=dt, eps=1e-8)
    brain = HierarchicalSemanticManifold(
        cfg, device,
        vocab=vocab,
        embed_dim=min(16, len(vocab)),
        chunk_min_len=2,
        chunk_max_len=4,
    )
    
    history: List[int] = []
    
    # Metrics tensors
    acc = torch.zeros(steps, dtype=torch.float32)
    energy = torch.zeros(steps, dtype=torch.float32)
    topo = torch.zeros(steps, dtype=torch.float32)
    chunks = torch.zeros(steps, dtype=torch.float32)
    ponder_shortcuts = torch.zeros(steps, dtype=torch.float32)
    ponder_dead_ends = torch.zeros(steps, dtype=torch.float32)
    ponder_hunger = torch.zeros(steps, dtype=torch.float32)
    
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
        
        # Dynamic system energy
        _exc = brain.attractors.get('excitation').abs().sum()
        _heat = brain.attractors.get('heat').abs().sum()
        _cexc = brain.chunks.excitation.abs().sum()
        _cheat = brain.chunks.heat.abs().sum()
        energy[t] = (_exc + _heat + _cexc + _cheat).detach().to(torch.float32).cpu()
        
        topo[t] = torch.tensor(
            _topological_entropy(brain.graph.src.detach(), brain.graph.w.detach(), eps=cfg.eps),
            dtype=torch.float32,
        )
        chunks[t] = float(brain.chunks.num_chunks)
        
        # Online learning
        brain.observe_next_token(nxt, probs=out.probs)
        
        # Idle pondering
        p = brain.idle_think(steps=1, dream_steps=context_len)
        ponder_shortcuts[t] = float(p.get("shortcuts", 0.0))
        ponder_dead_ends[t] = float(p.get("dead_ends", 0.0))
        ponder_hunger[t] = float(brain.hunger.mean().detach().cpu().item())
    
    # Rolling accuracy
    win = max(1, len(fwd))
    kern = torch.ones(win, dtype=torch.float32) / float(win)
    acc_smooth = torch.nn.functional.conv1d(
        acc.view(1, 1, -1), kern.view(1, 1, -1), padding=win // 2
    ).view(-1)[:steps]
    
    # Compute summary statistics
    pre_shift_acc = float(acc_smooth[shift_at - 100:shift_at].mean().item())
    post_shift_acc_immediate = float(acc_smooth[shift_at:shift_at + 50].mean().item())
    post_shift_acc_recovered = float(acc_smooth[shift_at + 200:shift_at + 300].mean().item()) if shift_at + 300 <= steps else float(acc_smooth[-100:].mean().item())
    
    # Find recovery point (first time accuracy returns to 80% of pre-shift)
    threshold = pre_shift_acc * 0.8
    recovery_step = None
    for t in range(shift_at, min(shift_at + 500, steps)):
        if float(acc_smooth[t].item()) >= threshold:
            recovery_step = t - shift_at
            break
    
    return {
        "config": asdict(cfg),
        "steps": steps,
        "shift_at": shift_at,
        "context_len": context_len,
        "vocab": vocab,
        "pre_shift_accuracy": pre_shift_acc,
        "post_shift_accuracy_immediate": post_shift_acc_immediate,
        "post_shift_accuracy_recovered": post_shift_acc_recovered,
        "recovery_steps": recovery_step,
        "final_chunks": int(chunks[-1].item()),
        "acc": acc.cpu().tolist(),
        "acc_smooth": acc_smooth.cpu().tolist(),
        "energy": energy.cpu().tolist(),
        "topo_entropy": topo.cpu().tolist(),
        "chunks": chunks.cpu().tolist(),
        "ponder_shortcuts": ponder_shortcuts.cpu().tolist(),
        "ponder_dead_ends": ponder_dead_ends.cpu().tolist(),
        "ponder_hunger_mean": ponder_hunger.cpu().tolist(),
    }


def generate_rule_shift_table(metrics: Dict[str, Any]) -> str:
    """Generate LaTeX table for rule shift results."""
    return r"""\begin{table}[t]
\centering
\caption{Rule-shift experiment results. The system adapts to a complete reversal of sequential structure at step """ + str(metrics.get('shift_at', 1000)) + r""".}
\label{tab:rule_shift}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Pre-shift accuracy & """ + f"{metrics.get('pre_shift_accuracy', 0):.1%}" + r""" \\
Post-shift accuracy (immediate) & """ + f"{metrics.get('post_shift_accuracy_immediate', 0):.1%}" + r""" \\
Post-shift accuracy (recovered) & """ + f"{metrics.get('post_shift_accuracy_recovered', 0):.1%}" + r""" \\
Steps to 80\% recovery & """ + f"{metrics.get('recovery_steps', 'N/A')}" + r""" \\
Final chunk count & """ + f"{metrics.get('final_chunks', 0)}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_rule_shift_figures(
    metrics: Dict[str, Any],
    figures_dir: Path,
) -> Dict[str, Path]:
    """Generate figures for rule shift experiment."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("  [WARN] matplotlib not available, skipping figures")
        return {}
    
    figures = {}
    steps = metrics["steps"]
    shift_at = metrics["shift_at"]
    x = list(range(steps))
    
    # Figure 1: Main rule shift dynamics
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax2 = ax1.twinx()
    
    # Top panel: accuracy and system metrics
    ax1.plot(x, metrics["acc_smooth"], 'b-', label="Accuracy (rolling)", linewidth=1.5)
    ax1.axvline(shift_at, color='r', linestyle='--', alpha=0.7, label="Rule shift")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Accuracy", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2.plot(x, metrics["energy"], 'g-', alpha=0.6, label="System energy", linewidth=1)
    ax2.plot(x, metrics["topo_entropy"], 'm-', alpha=0.6, label="Topo. entropy", linewidth=1)
    ax2.set_ylabel("Energy / Entropy", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # Bottom panel: pondering metrics
    ax3.plot(x, metrics["ponder_shortcuts"], 'c-', label="Shortcuts/step", linewidth=1)
    ax3.plot(x, metrics["ponder_dead_ends"], 'orange', label="Dead ends/step", linewidth=1)
    ax3.plot(x, metrics["ponder_hunger_mean"], 'purple', label="Hunger (mean)", linewidth=1)
    ax3.axvline(shift_at, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Pondering metrics")
    ax3.legend(loc='upper right', fontsize=8)
    
    fig.tight_layout()
    
    # Save as PDF
    fig_path = figures_dir / "rule_shift.pdf"
    fig.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    figures["rule_shift"] = fig_path
    print(f"  [FIGURE] rule_shift.pdf")
    
    # Figure 2: Pondering detail
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x, 0, metrics["ponder_shortcuts"], alpha=0.3, label="Shortcuts")
    ax.fill_between(x, 0, metrics["ponder_dead_ends"], alpha=0.3, label="Dead ends")
    ax.plot(x, metrics["ponder_hunger_mean"], 'k-', linewidth=1.5, label="Hunger")
    ax.axvline(shift_at, color='r', linestyle='--', alpha=0.7, label="Rule shift")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Count / Mean")
    ax.set_title("Idle Pondering Behavior")
    ax.legend(loc='upper right')
    fig2.tight_layout()
    
    fig_path2 = figures_dir / "pondering.pdf"
    fig2.savefig(fig_path2, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig2)
    figures["pondering"] = fig_path2
    print(f"  [FIGURE] pondering.pdf")
    
    return figures


def run_rule_shift_experiment(
    device: torch.device,
    tables_dir: Path,
    figures_dir: Path,
    steps: int = 2000,
    shift_at: int = 1000,
    context_len: int = 6,
    dt: float = 0.02,
):
    """
    Run the complete rule shift experiment and generate artifacts.
    """
    from .harness import ExperimentResult
    
    print("  Running rule shift simulation...")
    metrics = run_rule_shift(
        steps=steps,
        shift_at=shift_at,
        context_len=context_len,
        dt=dt,
        device=device,
    )
    
    print(f"  Pre-shift accuracy: {metrics['pre_shift_accuracy']:.1%}")
    print(f"  Post-shift (immediate): {metrics['post_shift_accuracy_immediate']:.1%}")
    print(f"  Post-shift (recovered): {metrics['post_shift_accuracy_recovered']:.1%}")
    if metrics['recovery_steps']:
        print(f"  Recovery steps: {metrics['recovery_steps']}")
    
    # Generate table
    table_content = generate_rule_shift_table(metrics)
    table_path = tables_dir / "rule_shift_summary.tex"
    table_path.write_text(table_content)
    print(f"  [TABLE] rule_shift_summary.tex")
    
    # Generate figures
    figures = generate_rule_shift_figures(metrics, figures_dir)
    
    # Save raw data as JSON
    json_path = figures_dir.parent / "rule_shift_data.json"
    # Only save summary, not full time series (too large)
    summary = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    json_path.write_text(json.dumps(summary, indent=2))
    
    return ExperimentResult(
        name="Rule Shift",
        metrics=metrics,
        tables={"rule_shift_summary": table_content},
        figures=figures,
    )
