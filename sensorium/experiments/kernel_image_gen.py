"""Kernel image handling via Universal Tokenizer (MNIST inpainting demo).

We demonstrate "native image handling" in the sense of the Universal Tokenizer:
pixels are bytes, hashed with position, with completion by sampling bytes.

Writes:
- `paper/tables/image_gen_summary.tex`
- `paper/figures/image_gen.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id


@dataclass(frozen=True, slots=True)
class KernelImageGenConfig:
    device: str = "mps"
    train_images: int = 60
    eval_images: int = 12
    hash_vocab_size: int = 4096
    mask_frac: float = 0.20


def _load_mnist_images(n_train: int, n_eval: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise ImportError("torchvision is required for kernel_image_gen") from e
    data_dir = Path("./data/mnist")
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    tr = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    train = []
    eval_ = []
    for i in range(min(n_train, len(tr))):
        img, _y = tr[i]
        train.append((img * 255.0).to(torch.uint8).view(-1))
    for i in range(min(n_eval, len(te))):
        img, _y = te[i]
        eval_.append((img * 255.0).to(torch.uint8).view(-1))
    return train, eval_


def run_kernel_image_gen(cfg: KernelImageGenConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    train_imgs, eval_imgs = _load_mnist_images(int(cfg.train_images), int(cfg.eval_images))
    eng_cfg = KernelEngineConfig(device=cfg.device, hash_vocab_size=int(cfg.hash_vocab_size))
    eng = KernelTokenEngine(eng_cfg)

    # "Train" by streaming a few full images (build carriers).
    for img in train_imgs:
        eng.reset()
        for p, bv in enumerate(img.tolist()):
            tid = hash_id(int(bv), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 64) == 0:
                eng.step(p)

    # Evaluate inpainting
    mse_sum = 0.0
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for img in eval_imgs:
        eng.reset()
        img_bytes = img.clone()
        n = int(img_bytes.numel())
        mask_n = int(float(cfg.mask_frac) * n)
        mask_idx = torch.randperm(n)[:mask_n]
        masked = img_bytes.clone()
        masked[mask_idx] = 0

        # Ingest observed bytes
        observed = torch.ones(n, dtype=torch.bool)
        observed[mask_idx] = False
        for p in range(n):
            if not bool(observed[p]):
                continue
            bv = int(masked[p].item())
            tid = hash_id(bv, p, hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 64) == 0:
                eng.step(p)

        # Fill missing sequentially
        recon = masked.clone()
        for p in mask_idx.tolist():
            b_pred, _scores = eng.predict_byte(int(p))
            recon[int(p)] = int(b_pred)
            tid = hash_id(int(b_pred), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            eng.step(int(p))

        # MSE in [0,1] space
        gt = img_bytes.to(torch.float32) / 255.0
        rr = recon.to(torch.float32) / 255.0
        mse = float(((gt - rr) ** 2).mean().item())
        mse_sum += mse
        if len(samples) < 6:
            samples.append((img_bytes.view(28, 28), masked.view(28, 28), recon.view(28, 28)))

    mse_avg = float(mse_sum / max(1, len(eval_imgs)))

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel MNIST inpainting via Universal Tokenizer byte completion.}
\label{tab:image_gen}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Reconstruction MSE & """ + f"{mse_avg:.4f}" + r""" \\
Mask fraction & """ + f"{float(cfg.mask_frac):.2f}" + r""" \\
Eval images & """ + f"{len(eval_imgs)}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "image_gen_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(samples), 3, figsize=(7, 2.2 * len(samples)))
        if len(samples) == 1:
            axes = axes.reshape(1, 3)
        for i, (gt, ms, rc) in enumerate(samples):
            for j, (im, title) in enumerate([(gt, "original"), (ms, "masked"), (rc, "recon")]):
                ax = axes[i, j]
                ax.imshow(im.numpy(), cmap="gray", vmin=0, vmax=255)
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(fdir / "image_gen.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {"reconstruction_mse": mse_avg}

