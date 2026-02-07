"""Run experiment datasets through the real manifold physics stack.

This module is intentionally explicit: by default it executes the Metal/MPS
thermodynamic + wave domains, and only uses analysis fallback when explicitly
enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch

from sensorium.kernels.runtime import get_device


@dataclass(frozen=True)
class ManifoldRunConfig:
    """Configuration for real manifold execution in experiments."""

    grid_size: tuple[int, int, int] = (64, 64, 64)
    max_steps: int = 16
    min_steps: int = 4
    init_heat: float = 0.1
    init_mass: float = 1.0
    init_energy_bias: float = 0.1
    position_jitter_fraction: float = 0.15
    velocity_scale: float = 0.0
    allow_analysis_fallback: bool = False
    analysis_mode_bins: int = 512
    omega_num_modes: int | None = None


def _collect_stream(
    stream: Iterable[Tuple[int, int]] | Iterator[Tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    byte_values: list[int] = []
    sequence_indices: list[int] = []
    sample_indices: list[int] = []

    prev_seq: int | None = None
    sample_idx = -1
    for byte_value, seq_idx in stream:
        seq_i = int(seq_idx)
        if prev_seq is None or seq_i <= prev_seq:
            sample_idx += 1
        prev_seq = seq_i

        byte_values.append(int(byte_value) & 0xFF)
        sequence_indices.append(seq_i)
        sample_indices.append(sample_idx)

    return (
        np.asarray(byte_values, dtype=np.int64),
        np.asarray(sequence_indices, dtype=np.int64),
        np.asarray(sample_indices, dtype=np.int64),
    )


def _empty_state(*, device: torch.device, grid_size: tuple[int, int, int]) -> dict:
    zeros_i64 = torch.zeros((0,), device=device, dtype=torch.int64)
    zeros_f32 = torch.zeros((0,), device=device, dtype=torch.float32)
    return {
        "byte_values": zeros_i64,
        "token_ids": zeros_i64,
        "sequence_indices": zeros_i64,
        "sample_indices": zeros_i64,
        "positions": torch.zeros((0, 3), device=device, dtype=torch.float32),
        "velocities": torch.zeros((0, 3), device=device, dtype=torch.float32),
        "masses": zeros_f32,
        "heats": zeros_f32,
        "energies": zeros_f32,
        "energy_osc": zeros_f32,
        "excitations": zeros_f32,
        "omega": zeros_f32,
        "phase": zeros_f32,
        "dx": float(1.0 / float(max(grid_size))),
        "grid_size": tuple(grid_size),
    }


def _bootstrap_state(
    byte_values: np.ndarray,
    sequence_indices: np.ndarray,
    sample_indices: np.ndarray,
    *,
    cfg: ManifoldRunConfig,
    device: torch.device,
) -> dict:
    n = int(byte_values.size)
    if n == 0:
        return _empty_state(device=device, grid_size=cfg.grid_size)

    gx, gy, gz = (int(cfg.grid_size[0]), int(cfg.grid_size[1]), int(cfg.grid_size[2]))
    gmax = int(max(gx, gy, gz))
    dx = float(1.0 / float(gmax))
    domain = np.asarray([gx * dx, gy * dx, gz * dx], dtype=np.float32)

    seq = sequence_indices.astype(np.int64, copy=False)
    byt = byte_values.astype(np.int64, copy=False)
    sam = sample_indices.astype(np.int64, copy=False)

    token_ids = ((seq << np.int64(8)) | (byt & np.int64(0xFF))).astype(np.int64)

    x = (seq % gx).astype(np.float32)
    y = ((seq // gx) % gy).astype(np.float32)
    z = ((seq // max(1, gx * gy)) % gz).astype(np.float32)

    base = np.stack(
        [
            (x + 0.5) * dx,
            (y + 0.5) * dx,
            (z + 0.5) * dx,
        ],
        axis=1,
    )

    # Keep particles inside their initial cell while avoiding exact overlap.
    jitter_frac = float(max(0.0, min(0.49, cfg.position_jitter_fraction)))
    h = (
        (seq.astype(np.uint64) * np.uint64(11400714819323198485))
        ^ (sam.astype(np.uint64) * np.uint64(7046029254386353131))
        ^ (byt.astype(np.uint64) * np.uint64(14029467366897019727))
    )
    jx = ((h & np.uint64(0x3FF)).astype(np.float32) / 1023.0) - 0.5
    jy = (((h >> np.uint64(10)) & np.uint64(0x3FF)).astype(np.float32) / 1023.0) - 0.5
    jz = (((h >> np.uint64(20)) & np.uint64(0x3FF)).astype(np.float32) / 1023.0) - 0.5
    jitter = np.stack([jx, jy, jz], axis=1).astype(np.float32)
    positions = np.mod(base + jitter * (jitter_frac * dx), domain[None, :]).astype(
        np.float32
    )

    v_scale = float(max(0.0, cfg.velocity_scale))
    if v_scale > 0.0:
        vx = (
            ((h >> np.uint64(4)) & np.uint64(0x3FF)).astype(np.float32) / 1023.0
        ) - 0.5
        vy = (
            ((h >> np.uint64(14)) & np.uint64(0x3FF)).astype(np.float32) / 1023.0
        ) - 0.5
        vz = (
            ((h >> np.uint64(24)) & np.uint64(0x3FF)).astype(np.float32) / 1023.0
        ) - 0.5
        velocities = np.stack([vx, vy, vz], axis=1).astype(np.float32) * v_scale
    else:
        velocities = np.zeros((n, 3), dtype=np.float32)

    phase = (
        (
            (seq.astype(np.float64) * 0.6180339887498948)
            + (sam.astype(np.float64) * 0.3819660112501052)
        )
        % 1.0
    ) * (2.0 * math.pi)
    phase = phase.astype(np.float32)

    omega = (0.1 + (byt.astype(np.float32) / 255.0) * 1.9).astype(np.float32)
    energies = (float(cfg.init_energy_bias) + (byt.astype(np.float32) / 255.0)).astype(
        np.float32
    )
    masses = np.full((n,), float(cfg.init_mass), dtype=np.float32)
    heats = np.full((n,), float(cfg.init_heat), dtype=np.float32)

    return {
        "byte_values": torch.from_numpy(byt).to(device=device, dtype=torch.int64),
        "token_ids": torch.from_numpy(token_ids).to(device=device, dtype=torch.int64),
        "sequence_indices": torch.from_numpy(seq).to(device=device, dtype=torch.int64),
        "sample_indices": torch.from_numpy(sam).to(device=device, dtype=torch.int64),
        "positions": torch.from_numpy(positions).to(device=device, dtype=torch.float32),
        "velocities": torch.from_numpy(velocities).to(
            device=device, dtype=torch.float32
        ),
        "masses": torch.from_numpy(masses).to(device=device, dtype=torch.float32),
        "heats": torch.from_numpy(heats).to(device=device, dtype=torch.float32),
        "energies": torch.from_numpy(energies).to(device=device, dtype=torch.float32),
        "energy_osc": torch.from_numpy(energies).to(device=device, dtype=torch.float32),
        "excitations": torch.from_numpy(omega).to(device=device, dtype=torch.float32),
        "omega": torch.from_numpy(omega).to(device=device, dtype=torch.float32),
        "phase": torch.from_numpy(phase).to(device=device, dtype=torch.float32),
        "dx": float(dx),
        "grid_size": tuple(cfg.grid_size),
    }


def run_stream_on_manifold(
    stream: Iterable[Tuple[int, int]] | Iterator[Tuple[int, int]],
    *,
    config: ManifoldRunConfig | None = None,
    on_step: Optional[Callable[[dict], None]] = None,
) -> tuple[dict, dict]:
    """Run an input stream through real manifold dynamics and return final state.

    Returns:
        (state, meta) where `meta` includes execution timing and termination info.
    """
    cfg = config or ManifoldRunConfig()
    byte_values, sequence_indices, sample_indices = _collect_stream(stream)
    n = int(byte_values.size)

    init_t0 = time.perf_counter()
    device_name = str(get_device())

    if device_name != "mps":
        if not bool(cfg.allow_analysis_fallback):
            raise RuntimeError(
                "Real manifold experiments require Metal/MPS. "
                "Set allow_analysis_fallback=True only for explicit analysis-mode runs."
            )
        from sensorium.experiments.state_builder import (
            StateBuildConfig,
            build_observation_state,
        )

        state = build_observation_state(
            zip(byte_values.tolist(), sequence_indices.tolist()),
            config=StateBuildConfig(
                grid_size=cfg.grid_size, mode_bins=int(cfg.analysis_mode_bins)
            ),
        )
        meta = {
            "run_backend": "analysis_fallback",
            "run_steps": 0,
            "run_termination": "analysis",
            "init_ms": float((time.perf_counter() - init_t0) * 1000.0),
            "simulate_ms": 0.0,
            "n_particles": int(n),
        }
        return state, meta

    from sensorium.kernels.metal.manifold_physics import (
        OmegaWaveDomain,
        ThermodynamicsDomain,
    )

    device = torch.device("mps")
    state = _bootstrap_state(
        byte_values=byte_values,
        sequence_indices=sequence_indices,
        sample_indices=sample_indices,
        cfg=cfg,
        device=device,
    )
    init_ms = float((time.perf_counter() - init_t0) * 1000.0)

    thermo = ThermodynamicsDomain(grid_size=cfg.grid_size)
    wave = OmegaWaveDomain(
        grid_size=cfg.grid_size,
        num_modes=(
            int(cfg.omega_num_modes)
            if isinstance(cfg.omega_num_modes, int) and cfg.omega_num_modes > 0
            else None
        ),
    )

    sim_t0 = time.perf_counter()
    step_count = 0
    termination = "budget"
    with torch.no_grad():
        for step in range(int(max(0, cfg.max_steps))):
            state = thermo.step(**state)
            state = wave.step(**state)
            step_count = int(step + 1)
            state["step"] = step_count
            if on_step is not None:
                try:
                    on_step(state)
                except Exception:
                    pass
            if step_count >= int(max(0, cfg.min_steps)) and bool(
                thermo.done_thinking(**state)
            ):
                termination = "quiet"
                break
        if hasattr(torch, "mps"):
            torch.mps.synchronize()
    simulate_ms = float((time.perf_counter() - sim_t0) * 1000.0)

    meta = {
        "run_backend": "mps",
        "run_steps": int(step_count),
        "run_termination": str(termination),
        "init_ms": float(init_ms),
        "simulate_ms": float(simulate_ms),
        "n_particles": int(n),
        "omega_num_modes": int(wave.num_modes),
    }
    return state, meta
