"""Kernel time-series forecasting (byte-quantized).

We quantize a synthetic time series to bytes and run next-byte prediction
using the same kernel engine as text.

Writes:
- `paper/tables/timeseries_summary.tex`
- `paper/figures/timeseries.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelTimeSeriesConfig:
    device: str = "mps"
    length: int = 3000
    steps: int = 800
    context_warmup: int = 64
    hash_vocab_size: int = 4096


def _make_series(n: int) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32)
    y = 0.001 * t + 0.5 * torch.sin(2 * torch.pi * t / 24.0) + 0.3 * torch.sin(2 * torch.pi * t / (24.0 * 7.0))
    y = y + 0.05 * torch.randn_like(y)
    return y


def _to_bytes(y: torch.Tensor) -> bytes:
    # Normalize to [0, 255]
    y = y.to(torch.float32)
    lo = float(y.min().item())
    hi = float(y.max().item())
    z = (y - lo) / max(1e-8, (hi - lo))
    b = torch.clamp((z * 255.0).round(), 0, 255).to(torch.uint8)
    return bytes(b.tolist())


def run_kernel_timeseries(cfg: KernelTimeSeriesConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    y = _make_series(int(cfg.length))
    data = _to_bytes(y)

    eng_cfg = KernelEngineConfig(device=cfg.device, hash_vocab_size=int(cfg.hash_vocab_size))
    eng = KernelTokenEngine(eng_cfg)
    eng.reset()

    # Warmup
    for i in range(int(cfg.context_warmup)):
        tid = hash_id(data[i], i, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        eng.step(i)

    correct = 0
    mae_sum = 0.0
    mse_sum = 0.0
    mae_ts: List[float] = []
    mse_ts: List[float] = []

    total = min(int(cfg.steps), len(data) - int(cfg.context_warmup) - 2)
    start = int(cfg.context_warmup)

    for t in range(total):
        pos = start + t
        true_b = int(data[pos])
        pred_b, _scores = eng.predict_byte(pos)
        if pred_b == true_b:
            correct += 1
        err = float(pred_b - true_b)
        mae_sum += abs(err)
        mse_sum += err * err
        mae_ts.append(mae_sum / (t + 1))
        mse_ts.append(mse_sum / (t + 1))

        tid = hash_id(true_b, pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        eng.step(pos)

    acc = float(correct / max(1, total))
    mae = float(mae_sum / max(1, total))
    mse = float(mse_sum / max(1, total))

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel time-series forecasting via next-byte prediction on a quantized synthetic series.}
\label{tab:timeseries}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Accuracy (exact byte) & """ + f"{acc:.3f}" + r""" \\
MAE (byte) & """ + f"{mae:.2f}" + r""" \\
MSE (byte) & """ + f"{mse:.2f}" + r""" \\
Steps & """ + f"{total}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "timeseries_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(total))
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        ax.plot(x, mse_ts, label="MSE (byte)", linewidth=1.3)
        ax.plot(x, mae_ts, label="MAE (byte)", linewidth=1.3)
        ax.set_xlabel("step")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / "timeseries.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {"accuracy": acc, "mae": mae, "mse": mse, "steps": total}

