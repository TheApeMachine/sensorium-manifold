"""Kernel text "diffusion" (byte denoising) via Universal Tokenizer.

We corrupt a byte sequence by masking bytes, then reconstruct by sequentially
sampling bytes using carrier scores.

Writes:
- `paper/tables/text_diffusion_summary.tex`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelTextDiffusionConfig:
    device: str = "mps"
    max_bytes: int = 1500
    mask_frac: float = 0.20
    hash_vocab_size: int = 4096


def _load_bytes() -> bytes:
    try:
        return Path("./paper/main.tex").read_bytes()
    except Exception:
        return b"Thermodynamic manifold diffusion experiment.\n" * 200


def run_kernel_text_diffusion(cfg: KernelTextDiffusionConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    data = bytearray(_load_bytes()[: int(cfg.max_bytes)])
    if len(data) < 64:
        raise RuntimeError("Not enough bytes for text diffusion demo.")

    eng_cfg = KernelEngineConfig(device=cfg.device, hash_vocab_size=int(cfg.hash_vocab_size))
    eng = KernelTokenEngine(eng_cfg)
    eng.reset()

    n = len(data)
    mask_n = int(float(cfg.mask_frac) * n)
    mask_idx = torch.randperm(n)[:mask_n].tolist()
    masked = bytearray(data)
    for i in mask_idx:
        masked[i] = 0  # NUL placeholder

    observed = [True] * n
    for i in mask_idx:
        observed[i] = False

    # Ingest observed bytes
    for pos in range(n):
        if not observed[pos]:
            continue
        tid = hash_id(masked[pos], pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        if (pos % 64) == 0:
            eng.step(pos)

    # Reconstruct masked bytes sequentially
    recon = bytearray(masked)
    correct = 0
    for pos in mask_idx:
        b_pred, _scores = eng.predict_byte(pos)
        recon[pos] = int(b_pred)
        if int(b_pred) == int(data[pos]):
            correct += 1
        tid = hash_id(int(b_pred), pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        eng.inject_id(tid)
        eng.step(pos)

    acc = float(correct / max(1, len(mask_idx)))

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel byte denoising (``text diffusion'') using the Universal Tokenizer.}
\label{tab:text_diffusion}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Masked-byte accuracy & """ + f"{acc:.3f}" + r""" \\
Mask fraction & """ + f"{float(cfg.mask_frac):.2f}" + r""" \\
Bytes & """ + f"{n}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "text_diffusion_summary.tex").write_text(table, encoding="utf-8")

    return {"masked_accuracy": acc}

