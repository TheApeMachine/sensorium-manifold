"""Kernel MNIST bytes classification (Universal Tokenizer).

We treat each image as a sequence of bytes; each (byte, position) hashes to an ID.
We train by streaming: [image_ids..., label_id]. Inference scores label IDs.

Writes:
- `paper/tables/mnist_bytes_summary.tex`
- `paper/figures/mnist_bytes.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .kernel_engine import KernelEngineConfig, KernelTokenEngine, hash_id, label_id


@dataclass(frozen=True, slots=True)
class KernelMNISTBytesConfig:
    device: str = "mps"
    train_samples: int = 300
    eval_samples: int = 100
    steps_per_sample: int = 1  # ingest cost is dominated by injections, not step count
    hash_vocab_size: int = 4096
    num_labels: int = 10


def _load_mnist(max_train: int, max_eval: int) -> tuple[List[torch.Tensor], List[int], List[torch.Tensor], List[int]]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise ImportError("torchvision is required for MNIST kernel experiments") from e

    data_dir = Path("./data/mnist")
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    tr = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    te = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)

    train_imgs: List[torch.Tensor] = []
    train_lbls: List[int] = []
    for i in range(min(max_train, len(tr))):
        img, y = tr[i]
        b = (img * 255.0).to(torch.uint8).view(-1)
        train_imgs.append(b)
        train_lbls.append(int(y))

    eval_imgs: List[torch.Tensor] = []
    eval_lbls: List[int] = []
    for i in range(min(max_eval, len(te))):
        img, y = te[i]
        b = (img * 255.0).to(torch.uint8).view(-1)
        eval_imgs.append(b)
        eval_lbls.append(int(y))

    return train_imgs, train_lbls, eval_imgs, eval_lbls


def run_kernel_mnist_bytes(cfg: KernelMNISTBytesConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    train_imgs, train_lbls, eval_imgs, eval_lbls = _load_mnist(int(cfg.train_samples), int(cfg.eval_samples))

    eng_cfg = KernelEngineConfig(
        device=cfg.device,
        hash_vocab_size=int(cfg.hash_vocab_size),
        num_labels=int(cfg.num_labels),
    )
    eng = KernelTokenEngine(eng_cfg)

    # Train: stream image bytes then label (online).
    for idx, (img, y) in enumerate(zip(train_imgs, train_lbls)):
        eng.reset()
        # image
        for p, bv in enumerate(img.tolist()):
            tid = hash_id(int(bv), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 64) == 0:
                eng.step(p)
        # observe label
        lid = label_id(int(y), special_size=eng_cfg.special_size, hash_vocab_size=eng_cfg.hash_vocab_size)
        eng.inject_id(lid, particles=eng_cfg.particles_per_token * 2, energy_scale=eng_cfg.energy_scale * 1.5)
        eng.step(int(idx))

    # Eval: score label IDs after streaming image.
    correct = 0
    conf = torch.zeros((cfg.num_labels, cfg.num_labels), dtype=torch.int32)
    for img, y in zip(eval_imgs, eval_lbls):
        eng.reset()
        for p, bv in enumerate(img.tolist()):
            tid = hash_id(int(bv), int(p), hash_vocab_size=eng_cfg.hash_vocab_size, hash_prime=eng_cfg.hash_prime, special_size=eng_cfg.special_size)
            eng.inject_id(tid)
            if (p % 64) == 0:
                eng.step(p)

        label_ids = torch.tensor(
            [label_id(i, special_size=eng_cfg.special_size, hash_vocab_size=eng_cfg.hash_vocab_size) for i in range(cfg.num_labels)],
            device=eng.dev,
            dtype=torch.long,
        )
        scores = eng.score_ids(label_ids)
        pred = int(torch.argmax(scores).item())
        conf[int(y), pred] += 1
        if pred == int(y):
            correct += 1

    acc = float(correct / max(1, len(eval_lbls)))

    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel MNIST classification from raw bytes using the Universal Tokenizer (position-aware hashing).}
\label{tab:mnist_bytes}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Accuracy & """ + f"{acc:.3f}" + r""" \\
Train samples & """ + f"{len(train_lbls)}" + r""" \\
Eval samples & """ + f"{len(eval_lbls)}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tdir / "mnist_bytes_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(conf.numpy(), cmap="Blues")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(f"MNIST confusion (acc={acc:.2f})")
        fig.tight_layout()
        fig.savefig(fdir / "mnist_bytes.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {"accuracy": acc}

