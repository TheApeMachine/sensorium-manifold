"""Helpers to build deterministic observation states from dataset streams.

These helpers intentionally avoid backend-specific physics execution so that
experiments can run in analysis mode (including CPU-only environments) while
still producing consistent map/path and SQL-observable structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import torch


@dataclass
class StateBuildConfig:
    """Configuration for deterministic state construction."""

    grid_size: tuple[int, int, int] = (64, 64, 64)
    mode_bins: int = 512


def build_observation_state(
    stream: Iterable[Tuple[int, int]] | Iterator[Tuple[int, int]],
    *,
    config: StateBuildConfig | None = None,
) -> dict:
    """Build a tensor state from a `(byte_value, sequence_index)` stream.

    The generated state contains the fields expected by map/path, wave, and SQL
    observers. It is deterministic and independent of GPU availability.
    """
    cfg = config or StateBuildConfig()
    gx, gy, gz = cfg.grid_size
    mode_bins = int(max(8, cfg.mode_bins))

    byte_values: list[int] = []
    sequence_indices: list[int] = []
    sample_indices: list[int] = []

    sample_idx = -1
    prev_seq: int | None = None

    for byte_value, seq_idx in stream:
        seq_i = int(seq_idx)
        if prev_seq is None or seq_i <= prev_seq:
            sample_idx += 1
        prev_seq = seq_i

        byte_values.append(int(byte_value) & 0xFF)
        sequence_indices.append(seq_i)
        sample_indices.append(sample_idx)

    n = len(byte_values)
    if n == 0:
        zeros = torch.zeros((0,), dtype=torch.float32)
        return {
            "byte_values": torch.zeros((0,), dtype=torch.int64),
            "token_ids": torch.zeros((0,), dtype=torch.int64),
            "sequence_indices": torch.zeros((0,), dtype=torch.int64),
            "sample_indices": torch.zeros((0,), dtype=torch.int64),
            "positions": torch.zeros((0, 3), dtype=torch.float32),
            "velocities": torch.zeros((0, 3), dtype=torch.float32),
            "masses": zeros,
            "heats": zeros,
            "energies": zeros,
            "excitations": zeros,
            "amplitudes": torch.zeros((mode_bins,), dtype=torch.float32),
            "mode_state": torch.zeros((mode_bins,), dtype=torch.int64),
            "omega": torch.zeros((mode_bins,), dtype=torch.float32),
            "phase": torch.zeros((mode_bins,), dtype=torch.float32),
            "psi_amplitude": torch.zeros((mode_bins,), dtype=torch.float32),
            "dx": float(1.0 / float(max(gx, gy, gz))),
            "grid_size": tuple(cfg.grid_size),
        }

    byte_np = np.asarray(byte_values, dtype=np.int64)
    seq_np = np.asarray(sequence_indices, dtype=np.int64)
    sample_np = np.asarray(sample_indices, dtype=np.int64)

    token_np = ((seq_np.astype(np.int64) << 8) | (byte_np.astype(np.int64) & 0xFF)).astype(np.int64)
    dx = float(1.0 / float(max(gx, gy, gz)))

    # Deterministic 3D placement from sequence index (shared across samples).
    x = (seq_np % gx).astype(np.float32)
    y = ((seq_np // gx) % gy).astype(np.float32)
    z = ((seq_np // max(1, gx * gy)) % gz).astype(np.float32)
    pos_np = np.stack(
        [
            (x + 0.5) * dx,
            (y + 0.5) * dx,
            (z + 0.5) * dx,
        ],
        axis=1,
    )

    vel_np = np.zeros((n, 3), dtype=np.float32)
    mass_np = np.ones((n,), dtype=np.float32)
    heat_np = np.full((n,), 0.1, dtype=np.float32)
    energy_np = (byte_np.astype(np.float32) / 255.0).astype(np.float32)
    excitation_np = (token_np % mode_bins).astype(np.float32)

    mode_ids = (token_np % mode_bins).astype(np.int64)
    mode_counts = np.bincount(mode_ids, minlength=mode_bins).astype(np.float32)
    psi_amp_np = np.sqrt(mode_counts).astype(np.float32)
    amp_np = psi_amp_np / max(1.0, float(np.max(psi_amp_np)))
    omega_np = np.linspace(0.0, 1.0, mode_bins, dtype=np.float32)
    phase_np = (2.0 * np.pi * omega_np).astype(np.float32)

    mode_state_np = np.zeros((mode_bins,), dtype=np.int64)
    mode_state_np[amp_np > 0.05] = 1
    mode_state_np[amp_np > 0.15] = 2

    return {
        "byte_values": torch.from_numpy(byte_np),
        "token_ids": torch.from_numpy(token_np),
        "sequence_indices": torch.from_numpy(seq_np),
        "sample_indices": torch.from_numpy(sample_np),
        "positions": torch.from_numpy(pos_np),
        "velocities": torch.from_numpy(vel_np),
        "masses": torch.from_numpy(mass_np),
        "heats": torch.from_numpy(heat_np),
        "energies": torch.from_numpy(energy_np),
        "excitations": torch.from_numpy(excitation_np),
        "amplitudes": torch.from_numpy(amp_np),
        "mode_state": torch.from_numpy(mode_state_np),
        "omega": torch.from_numpy(omega_np),
        "phase": torch.from_numpy(phase_np),
        "psi_amplitude": torch.from_numpy(psi_amp_np),
        "dx": dx,
        "grid_size": tuple(cfg.grid_size),
    }
