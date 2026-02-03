"""Kernel next-token (byte) prediction via Universal Tokenizer.

We treat text as raw UTF-8 bytes. At each position, we:
1) ingest context bytes (by injecting their hashed IDs)
2) score all 256 candidate next-bytes using carrier spectrum
3) take argmax as prediction, then "observe" by ingesting the true byte

Writes paper artifacts:
- `paper/tables/next_token_summary.tex`
- `paper/figures/next_token.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelNextTokenConfig:
    device: str = "mps"
    steps: int = 800
    context_warmup: int = 64  # bytes to warm up before scoring
    max_bytes: int = 4000
    omega_range: float = 2.0
    hash_vocab_size: int = 4096


def _load_text_bytes() -> bytes:
    # Default corpus: paper text (available locally; no network).
    try:
        p = Path("./paper/main.tex")
        return p.read_bytes()
    except Exception:
        return b"The cat sat on the mat.\n" * 200


def run_kernel_next_token(cfg: KernelNextTokenConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    data = _load_text_bytes()[: int(cfg.max_bytes)]
    if len(data) < (cfg.context_warmup + 2):
        raise RuntimeError("Not enough bytes for next-token experiment.")

    eng_cfg = KernelEngineConfig(
        device=cfg.device,
        omega_range=float(cfg.omega_range),
        hash_vocab_size=int(cfg.hash_vocab_size),
    )
    eng = KernelTokenEngine(eng_cfg)
    eng.reset()

    correct = 0
    nll_sum = 0.0
    acc_ts: List[float] = []
    nll_ts: List[float] = []

    # Warmup ingest
    for i in range(int(cfg.context_warmup)):
        tid = hash_id(data[i], i, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        eng.step(i)

    # Predict/observe loop
    start = int(cfg.context_warmup)
    total = min(int(cfg.steps), len(data) - start - 1)
    for t in range(total):
        pos = start + t
        true_b = int(data[pos])

        pred_b, scores = eng.predict_byte(pos)
        if pred_b == true_b:
            correct += 1

        # Convert scores -> probs via softmax for NLL.
        probs = torch.softmax(scores / 0.5, dim=0)
        nll = float((-torch.log(probs[true_b] + 1e-12)).detach().to("cpu").item())
        nll_sum += nll

        acc_ts.append(float(correct / (t + 1)))
        nll_ts.append(float(nll_sum / (t + 1)))

        # Observe: ingest true byte
        tid = hash_id(true_b, pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        eng.step(pos)

    acc = float(correct / max(1, total))
    avg_nll = float(nll_sum / max(1, total))
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel next-byte prediction using the Universal Tokenizer (UTF-8 bytes).}
\label{tab:next_token}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Accuracy & """ + f"{acc:.3f}" + r""" \\
Perplexity (softmax score) & """ + f"{ppl:.2f}" + r""" \\
Steps & """ + f"{total}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "next_token_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(total))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax1.plot(x, acc_ts, label="accuracy", linewidth=1.3)
        ax1.set_ylabel("accuracy")
        ax1.legend(loc="lower right", fontsize=8)
        ax2.plot(x, nll_ts, label="NLL", linewidth=1.3)
        ax2.set_ylabel("NLL")
        ax2.set_xlabel("step")
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / "next_token.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {"accuracy": acc, "ppl": ppl, "steps": total}

