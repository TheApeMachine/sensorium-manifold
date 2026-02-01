from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..core.config import PhysicsConfig
from ..core.diagnostics import SemanticDiagnosticsLogger
from ..core.viz import plot_pondering_jsonl
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
    tid = {t: i for i, t in enumerate(vocab)}

    # "The cat sat on the mat" with repeated 'the' to force >1-step memory.
    fwd = [
        tid["<bos>"],
        tid["the"],
        tid["cat"],
        tid["sat"],
        tid["on"],
        tid["the"],
        tid["mat"],
        tid["<eos>"],
    ]

    rev = [
        tid["<bos>"],
        tid["mat"],
        tid["the"],
        tid["on"],
        tid["sat"],
        tid["cat"],
        tid["the"],
        tid["<eos>"],
    ]

    return tid, fwd, rev


def run_rule_shift(
    *,
    steps: int,
    shift_at: int,
    context_len: int,
    dt: float,
    device: torch.device,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = ["<bos>", "the", "cat", "sat", "on", "mat", "<eos>"]
    tid, fwd, rev = _make_stream(vocab)

    # Pre-generate the full token stream (deterministic).
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
        cfg,
        device,
        vocab=vocab,
        embed_dim=min(16, len(vocab)),
        chunk_min_len=2,
        chunk_max_len=4,
    )

    # Pondering diagnostics (JSONL + optional CSV)
    ponder_jsonl = out_dir / "pondering.jsonl"
    ponder_csv = out_dir / "pondering.csv"
    brain.set_diagnostics(SemanticDiagnosticsLogger(csv_path=str(ponder_csv), jsonl_path=str(ponder_jsonl)))

    history: List[int] = []

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

        # Metrics
        # Dynamic system energy (exclude long-term structural mass)
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

        # Online structural update (metabolic shock).
        brain.observe_next_token(nxt, probs=out.probs)

        # Idle pondering: discover relations between what it already knows.
        p = brain.idle_think(steps=1, dream_steps=context_len)
        ponder_shortcuts[t] = float(p.get("shortcuts", 0.0))
        ponder_dead_ends[t] = float(p.get("dead_ends", 0.0))
        ponder_hunger[t] = float(brain.hunger.mean().detach().cpu().item())

    # Rolling accuracy for readability
    win = max(1, int(len(fwd)))
    kern = torch.ones(win, dtype=torch.float32) / float(win)
    acc_smooth = torch.nn.functional.conv1d(
        acc.view(1, 1, -1), kern.view(1, 1, -1), padding=win // 2
    ).view(-1)
    acc_smooth = acc_smooth[:steps]

    # Save raw metrics
    data = {
        "cfg": asdict(cfg),
        "steps": steps,
        "shift_at": shift_at,
        "context_len": context_len,
        "vocab": vocab,
        "tid": tid,
        "acc": acc.cpu().tolist(),
        "acc_smooth": acc_smooth.cpu().tolist(),
        "energy": energy.cpu().tolist(),
        "topo_entropy": topo.cpu().tolist(),
        "chunks": chunks.cpu().tolist(),
        "ponder_shortcuts": ponder_shortcuts.cpu().tolist(),
        "ponder_dead_ends": ponder_dead_ends.cpu().tolist(),
        "ponder_hunger_mean": ponder_hunger.cpu().tolist(),
    }
    json_path = out_dir / "rule_shift_metrics.json"
    json_path.write_text(__import__("json").dumps(data, indent=2))

    # Plot rule-shift + pondering
    try:
        import matplotlib.pyplot as plt

        x = list(range(steps))

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax2 = ax1.twinx()

        ax1.plot(x, acc_smooth.cpu().numpy(), label="accuracy (rolling)")
        ax1.axvline(shift_at, linestyle="--")
        ax1.set_ylim(0.0, 1.05)
        ax1.set_ylabel("accuracy")

        ax2.plot(x, energy.cpu().numpy(), label="system energy")
        ax2.plot(x, topo.cpu().numpy(), label="topology entropy")
        ax2.plot(x, chunks.cpu().numpy(), label="#chunks")
        ax2.set_ylabel("energy / entropy / count")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        ax3.plot(x, ponder_shortcuts.cpu().numpy(), label="shortcuts/step")
        ax3.plot(x, ponder_dead_ends.cpu().numpy(), label="dead_ends/step")
        ax3.plot(x, ponder_hunger.cpu().numpy(), label="hunger_mean")
        ax3.set_xlabel("time step")
        ax3.set_ylabel("pondering")
        ax3.legend(loc="upper left")

        fig.tight_layout()
        fig_path = out_dir / "rule_shift.png"
        fig.savefig(fig_path)
        plt.close(fig)

        # Also generate a dedicated pondering plot from JSONL (easy to read standalone).
        plot_pondering_jsonl(ponder_jsonl, out_dir / "pondering.png")
        return fig_path
    except Exception:
        # matplotlib not available or headless issues: return json path.
        return json_path


def main() -> None:
    p = argparse.ArgumentParser(description="Thermodynamic Manifold Rule-Shift benchmark")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--shift-at", type=int, default=1000)
    p.add_argument("--context-len", type=int, default=6)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--out-dir", type=str, default="./artifacts")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    path = run_rule_shift(
        steps=args.steps,
        shift_at=args.shift_at,
        context_len=args.context_len,
        dt=args.dt,
        device=device,
        out_dir=out_dir,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
