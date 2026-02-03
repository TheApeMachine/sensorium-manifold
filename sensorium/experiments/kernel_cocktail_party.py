"""Kernel cocktail party (two-speaker) separation experiment.

Uses `two_speakers.wav` bundled in this folder.

Mechanism (kernel stack):
1) Build carrier spectrum from a short prompt window A → select top-K carriers (mask A)
2) Build carrier spectrum from a short prompt window B → select top-K carriers (mask B)
3) Run autoregressive byte sampling on the overlap window using masked carrier scoring,
   producing two different reconstructions ("speaker A" and "speaker B").

Writes:
- `paper/tables/cocktail_party_summary.tex`
- `paper/figures/cocktail_party.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelCocktailPartyConfig:
    device: str = "mps"
    wav_path: Path = Path("sensorium/experiments/two_speakers.wav")

    # Windowing assumption (best-effort):
    prompt_seconds: float = 1.0
    prompt_a_start_s: float = 0.0
    prompt_b_start_s: float = 1.0
    overlap_start_s: float = 2.0
    overlap_seconds: float = 2.0

    # Byte modeling
    hash_vocab_size: int = 4096
    carrier_top_k: int = 16
    ingest_step_every: int = 256


def _read_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth}")

    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1)
    return int(sr), x


def _float_to_bytes(x: np.ndarray) -> np.ndarray:
    # [-1,1] -> [0,255]
    z = np.clip((x + 1.0) * 0.5, 0.0, 1.0)
    return np.clip(np.round(z * 255.0), 0, 255).astype(np.uint8)


def _bytes_to_float(b: np.ndarray) -> np.ndarray:
    z = b.astype(np.float32) / 255.0
    return (z * 2.0 - 1.0).astype(np.float32)


def _carrier_mask_from_engine(eng: KernelTokenEngine, top_k: int) -> torch.Tensor:
    st = eng._last_carrier_state
    if st is None:
        return torch.zeros((0,), device=eng.dev, dtype=torch.bool)
    amp = st["amplitudes"]
    m = int(amp.numel())
    if m == 0:
        return torch.zeros((0,), device=eng.dev, dtype=torch.bool)
    k = min(int(top_k), m)
    idx = torch.topk(amp, k=k, largest=True).indices
    mask = torch.zeros((m,), device=eng.dev, dtype=torch.bool)
    mask[idx] = True
    return mask


def _stft_mag(x: np.ndarray, *, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    # Torch STFT for consistency.
    t = torch.tensor(x, dtype=torch.float32)
    Z = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=n_fft, return_complex=True)
    mag = torch.abs(Z).numpy()
    return mag


def run_kernel_cocktail_party(cfg: KernelCocktailPartyConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    wav_path = Path(cfg.wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))

    sr, x = _read_wav_mono(wav_path)
    b = _float_to_bytes(x)

    def sl(start_s: float, dur_s: float) -> slice:
        i0 = int(start_s * sr)
        i1 = int((start_s + dur_s) * sr)
        i0 = max(0, min(i0, len(b)))
        i1 = max(0, min(i1, len(b)))
        return slice(i0, i1)

    a_sl = sl(cfg.prompt_a_start_s, cfg.prompt_seconds)
    b_sl = sl(cfg.prompt_b_start_s, cfg.prompt_seconds)
    o_sl = sl(cfg.overlap_start_s, cfg.overlap_seconds)

    prompt_a = b[a_sl]
    prompt_b = b[b_sl]
    overlap = b[o_sl]
    if len(overlap) < 256:
        raise RuntimeError("Overlap window too short in two_speakers.wav")

    eng_cfg = KernelEngineConfig(device=cfg.device, hash_vocab_size=int(cfg.hash_vocab_size), carrier_every=1)

    # Build A mask
    engA = KernelTokenEngine(eng_cfg)
    engA.reset()
    for i, bv in enumerate(prompt_a.tolist()):
        tid = hash_id(int(bv), int(i), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engA.inject_id(tid)
        if (i % int(cfg.ingest_step_every)) == 0:
            engA.step(i)
    engA.step(int(len(prompt_a)))
    maskA = _carrier_mask_from_engine(engA, top_k=int(cfg.carrier_top_k))

    # Build B mask
    engB = KernelTokenEngine(eng_cfg)
    engB.reset()
    for i, bv in enumerate(prompt_b.tolist()):
        tid = hash_id(int(bv), int(i), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engB.inject_id(tid)
        if (i % int(cfg.ingest_step_every)) == 0:
            engB.step(i)
    engB.step(int(len(prompt_b)))
    maskB = _carrier_mask_from_engine(engB, top_k=int(cfg.carrier_top_k))

    # Reconstruct overlap twice
    reconA = np.zeros_like(overlap)
    reconB = np.zeros_like(overlap)

    engA2 = KernelTokenEngine(eng_cfg)
    engA2.reset()
    # prime with prompt A
    for i, bv in enumerate(prompt_a.tolist()):
        tid = hash_id(int(bv), int(i), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engA2.inject_id(tid)
        if (i % int(cfg.ingest_step_every)) == 0:
            engA2.step(i)
    engA2.step(int(len(prompt_a)))
    # sample overlap bytes with maskA
    for t in range(int(len(overlap))):
        pos = int(t)  # local position
        bb, _scores = engA2.predict_byte(pos, carrier_mask=maskA)
        reconA[t] = bb
        tid = hash_id(int(bb), pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engA2.inject_id(tid)
        engA2.step(pos)

    engB2 = KernelTokenEngine(eng_cfg)
    engB2.reset()
    for i, bv in enumerate(prompt_b.tolist()):
        tid = hash_id(int(bv), int(i), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engB2.inject_id(tid)
        if (i % int(cfg.ingest_step_every)) == 0:
            engB2.step(i)
    engB2.step(int(len(prompt_b)))
    for t in range(int(len(overlap))):
        pos = int(t)
        bb, _scores = engB2.predict_byte(pos, carrier_mask=maskB)
        reconB[t] = bb
        tid = hash_id(int(bb), pos, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
        engB2.inject_id(tid)
        engB2.step(pos)

    # Metrics (best-effort, no ground-truth stems):
    mix = _bytes_to_float(overlap)
    a = _bytes_to_float(reconA)
    c = _bytes_to_float(reconB)
    eps = 1e-8
    rms_mix = float(np.sqrt(np.mean(mix * mix) + eps))
    rms_a = float(np.sqrt(np.mean(a * a) + eps))
    rms_c = float(np.sqrt(np.mean(c * c) + eps))
    scale = rms_mix / (rms_a + rms_c + eps)
    a_s = a * scale
    c_s = c * scale
    recon_sum = np.clip(a_s + c_s, -1.0, 1.0)
    recon_mse = float(np.mean((recon_sum - mix) ** 2))
    cos_ac = float(np.dot(a_s, c_s) / (np.linalg.norm(a_s) * np.linalg.norm(c_s) + eps))

    # Write artifacts
    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel cocktail-party separation (two-speaker mixture). Prompts build carrier masks; overlap is reconstructed twice using masked carrier scoring.}
\label{tab:cocktail_party}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Overlap length (s) & """ + f"{len(overlap)/sr:.2f}" + r""" \\
Top-K carriers per speaker & """ + f"{int(cfg.carrier_top_k)}" + r""" \\
Mixture reconstruction MSE (A+B) & """ + f"{recon_mse:.4f}" + r""" \\
Cosine similarity (A vs B) & """ + f"{cos_ac:.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "cocktail_party_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m_mix = _stft_mag(mix)
        m_a = _stft_mag(a_s)
        m_b = _stft_mag(c_s)

        def logimg(m):
            return np.log1p(m)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].imshow(logimg(m_mix), aspect="auto", origin="lower")
        axes[0].set_title("Mixture (overlap)")
        axes[1].imshow(logimg(m_a), aspect="auto", origin="lower")
        axes[1].set_title("Separated A (masked carriers)")
        axes[2].imshow(logimg(m_b), aspect="auto", origin="lower")
        axes[2].set_title("Separated B (masked carriers)")
        axes[2].set_xlabel("frame")
        for ax in axes:
            ax.set_ylabel("freq")
        fig.tight_layout()
        fig.savefig(fdir / "cocktail_party.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {
        "sr": sr,
        "recon_mse": recon_mse,
        "cos_ab": cos_ac,
        "overlap_seconds": float(len(overlap) / sr),
    }

