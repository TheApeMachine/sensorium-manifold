from __future__ import annotations

"""MNIST IDX dataset utilities.

Keeps MNIST parsing + stratified splits out of experiment files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


MNIST_IMAGE_SIZE = 28 * 28


def _read_u32_be(b: bytes, off: int) -> int:
    return int.from_bytes(b[off : off + 4], "big", signed=False)


def load_mnist_train(dir_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST train images+labels from IDX files in `dir_path`."""
    img_path = dir_path / "train-images-idx3-ubyte"
    lbl_path = dir_path / "train-labels-idx1-ubyte"
    if not img_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            "MNIST IDX files not found. Expected:\n"
            f"- {img_path}\n"
            f"- {lbl_path}\n"
        )

    img = img_path.read_bytes()
    lbl = lbl_path.read_bytes()

    # Images header: magic, count, rows, cols
    magic = _read_u32_be(img, 0)
    if magic != 2051:
        raise ValueError(f"Unexpected MNIST image magic {magic} (expected 2051)")
    n = _read_u32_be(img, 4)
    rows = _read_u32_be(img, 8)
    cols = _read_u32_be(img, 12)
    if rows != 28 or cols != 28:
        raise ValueError(f"Unexpected MNIST shape {rows}x{cols} (expected 28x28)")
    img_data = np.frombuffer(img, dtype=np.uint8, offset=16)
    if img_data.size != n * rows * cols:
        raise ValueError("MNIST image file size mismatch")
    images = img_data.reshape(n, rows * cols)

    # Labels header: magic, count
    magic_l = _read_u32_be(lbl, 0)
    if magic_l != 2049:
        raise ValueError(f"Unexpected MNIST label magic {magic_l} (expected 2049)")
    n_l = _read_u32_be(lbl, 4)
    if n_l != n:
        raise ValueError(f"MNIST labels count {n_l} != images count {n}")
    labels = np.frombuffer(lbl, dtype=np.uint8, offset=8)
    if labels.size != n:
        raise ValueError("MNIST label file size mismatch")

    return images, labels


@dataclass(frozen=True)
class MNISTSplit:
    train_images: np.ndarray  # (N, 784) uint8
    train_labels: np.ndarray  # (N,) uint8
    holdout_images: np.ndarray
    holdout_labels: np.ndarray


def split_holdout_stratified(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    holdout_per_digit: int,
    train_per_digit: int,
) -> MNISTSplit:
    """Hold out a fixed number per digit; optionally cap training per digit."""
    rng = np.random.RandomState(int(seed))
    holdout_idx = []
    train_idx = []
    for d in range(10):
        idx = np.where(labels == d)[0]
        rng.shuffle(idx)
        h = int(holdout_per_digit)
        if h <= 0:
            raise ValueError("holdout_per_digit must be > 0")
        hold = idx[:h]
        rest = idx[h:]
        holdout_idx.append(hold)

        tpd = int(train_per_digit)
        if tpd > 0:
            rest = rest[:tpd]
        train_idx.append(rest)

    holdout_idx = np.concatenate(holdout_idx)
    train_idx = np.concatenate(train_idx)
    rng.shuffle(train_idx)
    rng.shuffle(holdout_idx)

    return MNISTSplit(
        train_images=images[train_idx],
        train_labels=labels[train_idx],
        holdout_images=images[holdout_idx],
        holdout_labels=labels[holdout_idx],
    )


def iter_images(images: np.ndarray) -> Iterator[bytes]:
    """Yield raw image bytes (784 bytes/image)."""
    for row in images:
        yield row.tobytes()

