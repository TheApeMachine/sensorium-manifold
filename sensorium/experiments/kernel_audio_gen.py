"""Kernel audio handling via Universal Tokenizer (waveform inpainting demo).

We quantize waveform samples to bytes and reconstruct a missing segment by
sampling bytes, similar to the MNIST inpainting experiment.

Writes:
- `paper/tables/audio_gen_summary.tex`
- `paper/figures/audio_gen.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelAudioGenConfig:
    device: str = "mps"
    length: int = 4000
    train_samples: int = 40
    eval_samples: int = 10
    hash_vocab_size: int = 4096
    mask_frac: float = 0.20


def _synthetic_audio(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    t = torch.linspace(0, 1.0, n)
    f1 = 220.0 + float(torch.randint(0, 440, (1,), generator=g).item())
    f2 = f1 * 2.0
    f3 = f1 * 1.5
    y = 0.5 * torch.sin(2 * torch.pi * f1 * t) + 0.3 * torch.sin(2 * torch.pi * f2 * t) + 0.2 * torch.sin(2 * torch.pi * f3 * t)
    env = torch.exp(-3.0 * t)
    y = y * env
    y = y / (y.abs().max() + 1e-8)
    return y.to(torch.float32)


def _to_bytes(y: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,255]
    z = (y.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (z * 255.0).round().clamp(0, 255).to(torch.uint8)


def _from_bytes(b: torch.Tensor) -> torch.Tensor:
    z = b.to(torch.float32) / 255.0
    return (z * 2.0 - 1.0).to(torch.float32)


def run_kernel_audio_gen(cfg: KernelAudioGenConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    train = [_to_bytes(_synthetic_audio(int(cfg.length), seed=i)) for i in range(int(cfg.train_samples))]
    eval_ = [_to_bytes(_synthetic_audio(int(cfg.length), seed=1000 + i)) for i in range(int(cfg.eval_samples))]

    eng_cfg = KernelEngineConfig(device=cfg.device, hash_vocab_size=int(cfg.hash_vocab_size))
    eng = KernelTokenEngine(eng_cfg)

    # Train by streaming a few samples (carriers build up).
    for wav in train:
        eng.reset()
        for p, bv in enumerate(wav.tolist()):
            tid = hash_id(int(bv), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 128) == 0:
                eng.step(p)

    mse_sum = 0.0
    demo: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for wav in eval_:
        eng.reset()
        n = int(wav.numel())
        mask_n = int(float(cfg.mask_frac) * n)
        mask_idx = torch.randperm(n)[:mask_n]
        masked = wav.clone()
        masked[mask_idx] = 128  # mid-level

        observed = torch.ones(n, dtype=torch.bool)
        observed[mask_idx] = False
        for p in range(n):
            if not bool(observed[p]):
                continue
            bv = int(masked[p].item())
            tid = hash_id(bv, p, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 128) == 0:
                eng.step(p)

        recon = masked.clone()
        for p in mask_idx.tolist():
            b_pred, _scores = eng.predict_byte(int(p))
            recon[int(p)] = int(b_pred)
            tid = hash_id(int(b_pred), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            eng.step(int(p))

        gt = _from_bytes(wav)
        rr = _from_bytes(recon)
        mse = float(((gt - rr) ** 2).mean().item())
        mse_sum += mse
        if len(demo) < 3:
            demo.append((gt, _from_bytes(masked), rr))

    mse_avg = float(mse_sum / max(1, len(eval_)))

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel audio inpainting via Universal Tokenizer byte completion (synthetic audio).}
\label{tab:audio_gen}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Waveform MSE & """ + f"{mse_avg:.4f}" + r""" \\
Mask fraction & """ + f"{float(cfg.mask_frac):.2f}" + r""" \\
Eval samples & """ + f"{len(eval_)}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "audio_gen_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(demo), 1, figsize=(9, 2.2 * len(demo)))
        if len(demo) == 1:
            axes = [axes]
        for i, (gt, ms, rc) in enumerate(demo):
            ax = axes[i]
            ax.plot(gt.numpy(), label="gt", linewidth=1.0, alpha=0.8)
            ax.plot(ms.numpy(), label="masked", linewidth=1.0, alpha=0.6)
            ax.plot(rc.numpy(), label="recon", linewidth=1.0, alpha=0.8)
            ax.set_xlim(0, int(cfg.length))
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / "audio_gen.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {"mse": mse_avg}

