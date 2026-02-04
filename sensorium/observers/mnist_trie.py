from __future__ import annotations

"""MNIST trie metrics, decoding, and paper-ready visualizations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


MNIST_IMAGE_SIZE = 28 * 28


def energy_by_token_id(token_ids_t: torch.Tensor, energies_t: torch.Tensor, *, vocab: int) -> np.ndarray:
    """Aggregate energy per token id from manifold state."""
    tid = token_ids_t.detach().to("cpu").to(torch.int64).numpy()
    ene = energies_t.detach().to("cpu").to(torch.float32).numpy()
    out = np.zeros((int(vocab),), dtype=np.float64)
    out += np.bincount(tid, weights=ene, minlength=int(vocab))
    return out


def unique_token_counts_per_pos(images: np.ndarray, *, prime: int, vocab: int) -> np.ndarray:
    """For each pixel position, count unique token_ids across images."""
    mask = int(vocab) - 1
    pos = np.arange(MNIST_IMAGE_SIZE, dtype=np.int64)
    tids = (images.astype(np.int64) * int(prime) + pos[None, :]) & mask
    uniq = np.empty((MNIST_IMAGE_SIZE,), dtype=np.int32)
    for p in range(MNIST_IMAGE_SIZE):
        uniq[p] = np.unique(tids[:, p]).size
    return uniq.reshape(28, 28)


def decode_by_energy(
    energy_tid: np.ndarray,
    *,
    prime: int,
    vocab: int,
    prompt_flat: np.ndarray,
    prompt_len: int,
) -> np.ndarray:
    """Decode missing pixels by argmax over bytes using learned token energies."""
    mask = int(vocab) - 1
    recon = prompt_flat.copy().astype(np.uint8)
    for pos in range(int(prompt_len), MNIST_IMAGE_SIZE):
        best_b = 0
        best_e = -1.0
        for b in range(256):
            tid = ((b * int(prime)) + int(pos)) & mask
            e = float(energy_tid[tid])
            if e > best_e:
                best_e = e
                best_b = b
        recon[pos] = np.uint8(best_b)
    return recon.reshape(28, 28)


def reconstruct_by_particle_selection(
    token_ids_t: torch.Tensor,
    energies_t: torch.Tensor,
    excitations_t: torch.Tensor,
    *,
    train_images: int,
    prompt_image_index: int,
    prompt_flat: np.ndarray,
    prompt_len: int,
    prime: int,
    vocab: int,
    score: str = "energy",
    enforce_prompt: bool = False,
    retrieve_n: int = MNIST_IMAGE_SIZE,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Reconstruct a 28x28 image by selecting particles (tokens) from the manifold.

    This matches the "inference as observation" framing:
    - ingest train stream (+ prompts appended)
    - let dynamics settle
    - for each pixel position, pick the *most salient* particle among training images
      (optionally conditioned by the prompt's known region)
    - dehash selected token_ids back to bytes and reshape.

    Notes:
    - Assumes tokenization used `segment_size = 784` and each image chunk is 784 bytes,
      so training tokens are `train_images * 784` and position is the column index.
    - `prompt_image_index` is the index within the appended prompt block (0-based).
    """
    if int(vocab) <= 0 or (int(vocab) & (int(vocab) - 1)) != 0:
        raise ValueError("vocab must be a power-of-two")
    mask = int(vocab) - 1
    inv_prime = pow(int(prime), -1, int(vocab))

    if prompt_flat.dtype != np.uint8:
        prompt_flat = prompt_flat.astype(np.uint8)
    if prompt_flat.size != MNIST_IMAGE_SIZE:
        raise ValueError(f"prompt_flat must have length {MNIST_IMAGE_SIZE}, got {prompt_flat.size}")

    device = token_ids_t.device
    train_images_i = int(train_images)
    train_end = train_images_i * MNIST_IMAGE_SIZE
    if train_end <= 0:
        raise ValueError("train_images must be > 0")
    if token_ids_t.numel() < train_end:
        raise ValueError("State has fewer tokens than expected training tokens")

    # Reshape training tokens as (train_images, 784) so we can pick per-position argmax.
    tid_train = token_ids_t[:train_end].to(torch.int64).view(train_images_i, MNIST_IMAGE_SIZE)
    ene_train = energies_t[:train_end].to(torch.float32).view(train_images_i, MNIST_IMAGE_SIZE)
    exc_train = excitations_t[:train_end].to(torch.float32).view(train_images_i, MNIST_IMAGE_SIZE)

    # Prompt segment is appended after training images.
    prompt_start = train_end + int(prompt_image_index) * MNIST_IMAGE_SIZE
    prompt_stop = prompt_start + MNIST_IMAGE_SIZE
    if token_ids_t.numel() < prompt_stop:
        raise ValueError("State does not include the requested prompt segment")

    # Condition on the prompt's known region by building a simple excitation "signature".
    # Weight known pixels by intensity so we focus on strokes, not background.
    w_np = (prompt_flat[: int(prompt_len)].astype(np.float32) / 255.0)
    if w_np.size == 0:
        w_np = np.ones((1,), dtype=np.float32)
    w = torch.tensor(w_np, device=device, dtype=torch.float32)
    w = w + 1e-3  # avoid zero weights

    exc_prompt = excitations_t[prompt_start : prompt_start + int(prompt_len)].to(torch.float32)
    if exc_prompt.numel() != w.numel():
        # If prompt_len is out of range, fall back to using full prompt.
        w = torch.ones((exc_prompt.numel(),), device=device, dtype=torch.float32)

    w_sum = w.sum()
    mu = (w * exc_prompt).sum() / torch.clamp(w_sum, min=1e-6)
    var = (w * (exc_prompt - mu).pow(2)).sum() / torch.clamp(w_sum, min=1e-6)
    sigma = torch.sqrt(torch.clamp(var, min=1e-6))

    # Compute a conditioned score for every training particle.
    # The idea: particles whose excitation matches the prompt signature get boosted.
    if score == "energy":
        base = ene_train
    elif score == "heat":
        # heat isn't part of the inputs here; caller would need to pass it.
        raise ValueError("score='heat' not supported in this function (pass energies_t or extend signature)")
    else:
        raise ValueError(f"Unknown score mode: {score!r}")

    z = (exc_train - mu) / sigma
    gate = torch.exp(-0.5 * z * z)
    s = base * gate

    # For each position (column), pick the best training image index.
    best_img = torch.argmax(s, dim=0)  # (784,)
    best_score = torch.amax(s, dim=0)  # (784,)
    cols = torch.arange(MNIST_IMAGE_SIZE, device=device, dtype=torch.int64)
    chosen_tid = tid_train[best_img, cols]  # (784,)

    # Dehash back to bytes for each pixel position.
    pos = cols
    target = (chosen_tid - pos) & mask
    recovered = (target * int(inv_prime)) & mask

    # Map to uint8: keep only valid bytes (<256), otherwise zero.
    rec = recovered.to(torch.int64)
    out = torch.zeros((MNIST_IMAGE_SIZE,), device=device, dtype=torch.uint8)
    ok = rec < 256
    out[ok] = rec[ok].to(torch.uint8)

    # Optionally enforce the prompt on the known region (copy exact bytes).
    if bool(enforce_prompt) and int(prompt_len) > 0:
        out[: int(prompt_len)] = torch.tensor(prompt_flat[: int(prompt_len)], device=device, dtype=torch.uint8)

    # Optionally retrieve fewer than 784 particles (e.g. 783) by dropping the lowest-score positions.
    # This keeps output length 784 by leaving dropped positions as zeros (or prompt if enforced).
    rn = int(retrieve_n)
    dropped = None
    if rn < MNIST_IMAGE_SIZE:
        # keep top-rn scores
        keep = torch.topk(best_score, k=max(1, rn)).indices
        keep_mask = torch.zeros((MNIST_IMAGE_SIZE,), device=device, dtype=torch.bool)
        keep_mask[keep] = True
        dropped = (~keep_mask).detach().to("cpu").numpy()
        out = torch.where(keep_mask, out, torch.zeros_like(out))

    recon = out.detach().to("cpu").numpy().reshape(28, 28)
    if not return_debug:
        return recon

    dbg = {
        "best_img": best_img.detach().to("cpu").numpy().reshape(28, 28),
        "best_score": best_score.detach().to("cpu").numpy().reshape(28, 28),
        "chosen_tid": chosen_tid.detach().to("cpu").numpy().reshape(28, 28),
        "dropped_mask": dropped.reshape(28, 28) if dropped is not None else None,
        "mu": float(mu.detach().item()),
        "sigma": float(sigma.detach().item()),
    }
    return recon, dbg


def render_selection_debug(
    *,
    artifact_path: callable,
    title: str,
    target: np.ndarray,
    prompt_img: np.ndarray,
    recon: np.ndarray,
    dbg: dict,
    train_mean: np.ndarray | None = None,
    train_max: np.ndarray | None = None,
) -> Path:
    """Big diagnostic figure to reason about inference choices."""
    import matplotlib.pyplot as plt

    path = artifact_path("figures", "mnist_trie_selection_debug.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    best_img = dbg.get("best_img")
    best_score = dbg.get("best_score")
    dropped = dbg.get("dropped_mask")

    err = np.abs(target.astype(np.float32) - recon.astype(np.float32))

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.25)

    def _show(ax, img, t, cmap="gray", vmin=0, vmax=255):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])

    ax0 = fig.add_subplot(gs[0, 0])
    _show(ax0, target, "Target (held-out)", cmap="gray")

    ax1 = fig.add_subplot(gs[0, 1])
    _show(ax1, prompt_img, "Query shown to model", cmap="gray")

    ax2 = fig.add_subplot(gs[0, 2])
    _show(ax2, recon, "Reconstruction (selected particles)", cmap="gray")

    ax3 = fig.add_subplot(gs[0, 3])
    _show(ax3, err, "Abs error", cmap="magma", vmin=0, vmax=255)

    ax4 = fig.add_subplot(gs[0, 4])
    if train_max is not None:
        _show(ax4, train_max, "Train per-pixel MAX", cmap="gray")
    elif train_mean is not None:
        _show(ax4, train_mean, "Train per-pixel MEAN", cmap="gray")
    else:
        ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 0])
    if best_img is not None:
        _show(ax5, best_img, "Selected training image index", cmap="viridis", vmin=0, vmax=max(1, int(np.nanmax(best_img))))
    else:
        ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 1])
    if best_score is not None:
        _show(ax6, best_score, "Best score per pixel", cmap="inferno", vmin=float(np.nanmin(best_score)), vmax=float(np.nanmax(best_score)))
    else:
        ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    if dropped is not None:
        _show(ax7, dropped.astype(np.float32), "Dropped positions (retrieve_n<784)", cmap="gray", vmin=0, vmax=1)
    else:
        ax7.axis("off")

    ax8 = fig.add_subplot(gs[1, 3])
    if train_mean is not None:
        _show(ax8, train_mean, "Train per-pixel MEAN", cmap="gray")
    else:
        ax8.axis("off")

    ax9 = fig.add_subplot(gs[1, 4])
    ax9.axis("off")
    mu = dbg.get("mu")
    sigma = dbg.get("sigma")
    ax9.text(
        0.02,
        0.98,
        "\n".join(
            [
                title,
                "",
                f"mu(exc_prompt)={mu:.4g}" if mu is not None else "",
                f"sigma(exc_prompt)={sigma:.4g}" if sigma is not None else "",
                "",
                "Hypothesis check:",
                "- if recon ~= train MAX, you're retrieving a global 'white union'",
                "- if best_score saturates, query conditioning is too weak",
                "",
                "Tip: if best_img is nearly constant,",
                "your scoring collapses to a single memorized exemplar.",
            ]
        ),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    fig.suptitle("Thermodynamic Trie — particle selection diagnostics", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


@dataclass(frozen=True)
class RecallMetrics:
    train_images: int
    holdout_images: int
    prompt_rows: int
    prompt_len: int
    completion_exact_acc: float
    completion_mae_0_1: float
    mean_unique_per_pixel: float
    max_unique_per_pixel: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "train_images": self.train_images,
            "holdout_images": self.holdout_images,
            "prompt_rows": self.prompt_rows,
            "prompt_len": self.prompt_len,
            "completion_exact_acc": self.completion_exact_acc,
            "completion_mae_0_1": self.completion_mae_0_1,
            "mean_unique_per_pixel": self.mean_unique_per_pixel,
            "max_unique_per_pixel": self.max_unique_per_pixel,
        }


def render_hero(
    *,
    artifact_path: callable,
    uniq_counts: np.ndarray,
    target: np.ndarray,
    prompt_img: np.ndarray,
    recon: np.ndarray,
    metrics: RecallMetrics,
) -> Path:
    import matplotlib.pyplot as plt

    path = artifact_path("figures", "mnist_trie_hero.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(uniq_counts, cmap="magma", interpolation="nearest")
    ax0.set_title("Bifurcation heatmap\n#unique addresses per pixel position")
    ax0.set_xticks([])
    ax0.set_yticks([])
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    per_row = uniq_counts.reshape(28, 28).mean(axis=1)
    ax1.plot(np.arange(28), per_row, "o-", linewidth=2)
    ax1.set_title("Bifurcation by row\n(mean unique addresses)")
    ax1.set_xlabel("row")
    ax1.set_ylabel("mean unique token_ids")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    m = metrics.as_dict()
    txt = []
    txt.append("MNIST Trie Recall Summary")
    txt.append("=" * 28)
    txt.append(f"train_images: {m['train_images']}")
    txt.append(f"holdout_images: {m['holdout_images']}")
    txt.append(f"prompt_rows: {m['prompt_rows']}  (len={m['prompt_len']})")
    txt.append("")
    txt.append("Completion metrics (masked region only):")
    txt.append(f"- exact_acc: {m['completion_exact_acc']:.3f}")
    txt.append(f"- mae_0_1:   {m['completion_mae_0_1']:.3f}")
    txt.append("")
    txt.append("Compression:")
    txt.append(f"- mean unique per pixel: {m['mean_unique_per_pixel']:.1f}")
    txt.append(f"- max unique per pixel:  {m['max_unique_per_pixel']}")
    ax2.text(
        0.02,
        0.98,
        "\n".join(txt),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    def _show(ax, img, title):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    _show(ax3, target, "Held-out target")

    ax4 = fig.add_subplot(gs[1, 1])
    _show(ax4, prompt_img, "Prompt (masked)")

    ax5 = fig.add_subplot(gs[1, 2])
    _show(ax5, recon, "Reconstruction from trie")

    fig.suptitle(
        "Compressed Thermodynamic Trie on MNIST\nHoldout reconstruction from content-addressable space",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def render_inference_grid(
    *,
    artifact_path: callable,
    labels: np.ndarray,
    targets: np.ndarray,
    prompts: np.ndarray,
    recons: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path = artifact_path("figures", "mnist_trie_inference_grid.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(targets.shape[0])
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 0.9 * n + 1.5))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        for j, (img, title) in enumerate(
            [
                (targets[i], f"target (y={int(labels[i])})"),
                (prompts[i], "prompt"),
                (recons[i], "recon"),
            ]
        ):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0 and i > 0:
                ax.set_ylabel(str(int(labels[i])), rotation=0, labelpad=12, fontsize=10, va="center")

    fig.suptitle("MNIST holdout reconstruction (one per digit)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

