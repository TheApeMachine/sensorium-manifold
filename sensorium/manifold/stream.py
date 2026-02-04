"""Stream (byte-chunk) manifold.

This is the minimal "generator → spectral carriers → state" manifold used by the
kernel experiments in `sensorium/experiments/`.

Key properties:
- Input is a generator/iterator yielding raw byte chunks (or (meta, bytes)).
- We hash each yielded chunk into an ID and seed an oscillator ensemble.
- We run the spectral carrier physics until a readiness policy is satisfied.
- If `dashboard=True`, we start the animated `SimulationDashboard`, record it to
  a video, and cleanly shut it down after the run.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch


class PrefetchIterator:
    """Wraps an iterator with a background thread that pre-fills a queue."""

    _SENTINEL = object()

    def __init__(self, iterable: Iterable, maxsize: int = 4096):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._fill, args=(iter(iterable),), daemon=True)
        self._thread.start()

    def _fill(self, it: Iterator) -> None:
        # Background prefetch should never crash the main process with a noisy
        # traceback. If the source iterator errors (e.g. missing dataset files),
        # we stop prefetching and let the consumer see an empty stream.
        try:
            for item in it:
                self._queue.put(item)
        except Exception:
            # Best-effort: swallow generator errors to keep experiments robust
            # in "missing optional dataset" scenarios.
            pass
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item

from optimizer.metal.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierConfig,
    SpectralCarrierPhysics,
)

from .carriers import CarrierState
from .visualizer import SimulationDashboard

ChunkPayload = Union[int, bytes, bytearray, memoryview]
ChunkItem = Union[ChunkPayload, Tuple[Any, ChunkPayload]]
ChunkSource = Union[Iterable[ChunkItem], Callable[[], Iterable[ChunkItem]]]
ReadyCondition = Union[str, Callable[["Manifold", dict[str, torch.Tensor], int], bool]]


@dataclass
class ManifoldConfig:
    """Minimal config: bytes in, readiness out."""

    generator: Optional[ChunkSource] = None

    # Fast path: provide raw bytes and segment size directly (skips generator overhead)
    # bytes_data: raw byte array (numpy uint8 or bytes)
    # segment_size: position resets every segment_size bytes (e.g., 784 for MNIST images)
    bytes_data: Optional[np.ndarray] = None
    segment_size: Optional[int] = None

    # Readiness policy:
    # - "crystallized": stop when any carrier reaches state=2
    # - "thinking_complete": stop when convergence heuristics pass (low conflict, enough carriers, etc.)
    # - callable: user-supplied predicate (manifold, carrier_out, step) -> bool
    ready: ReadyCondition = "crystallized"

    # Convenience knob used by some experiments (the generator decides how to chunk;
    # the manifold itself doesn't interpret chunk_size).
    chunk_size: int = 1

    # ------------------------------------------------------------------
    # Universal Tokenizer / hashing
    # ------------------------------------------------------------------
    # We treat (byte, index) -> token_id via a simple hash. Collision rate is
    # controlled by `hash_vocab_size` and is the primary stress knob for the
    # "collision is compression" experiments.
    #
    # NOTE: `hash_vocab_size` is required to be a power-of-two so we can use
    # a bitmask fast-path on GPU.
    hash_prime: int = 31
    hash_vocab_size: int = 4096

    # Dashboard / recording
    dashboard: bool = False
    dashboard_video_path: Path | None = None
    dashboard_video_fps: int = 30
    dashboard_update_interval: int = 10

    # ------------------------------------------------------------------
    # Performance / scale
    # ------------------------------------------------------------------
    # If False, we run *spectral carriers only* and do not allocate particle
    # positions/velocities/mass/heat or run spatial physics/collisions.
    enable_spatial_physics: bool = True

    # Whether to keep `token_ids` in the returned state (can be huge at scale).
    store_token_ids: bool = True

    # Parameters for built-in "thinking_complete" readiness policy.
    ready_min_carriers: int = 1
    ready_max_conflict: float = 0.20
    ready_min_crystallized_frac: float = 0.0

    # Fixed budget to prevent runaway loops.
    max_steps: int = 5000

    # Logging
    verbose: bool = True
    log_every: int = 10


def _normalize_chunk(x: ChunkItem) -> Tuple[Optional[Any], bytes]:
    """Normalize a generator yield into (meta, payload_bytes)."""
    meta: Optional[Any] = None
    payload: Any = x

    if isinstance(x, tuple) and len(x) == 2:
        meta, payload = x[0], x[1]

    if isinstance(payload, int):
        return meta, bytes([int(payload) & 0xFF])

    if isinstance(payload, (bytes, bytearray, memoryview)):
        return meta, bytes(payload)

    raise TypeError(f"generator yielded unsupported payload type: {type(payload)!r}")


def _iter_chunks(src: ChunkSource) -> Iterator[Tuple[Optional[Any], bytes]]:
    it = src() if callable(src) else src
    for x in it:
        meta, payload = _normalize_chunk(x)
        if len(payload) == 0:
            # Ignore empty yields (common in file streaming loops).
            continue
        yield meta, payload


def _default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("No supported backend available (need MPS or CUDA)")


class Manifold:
    """A tiny wrapper around `SpectralCarrierPhysics` for byte streams."""

    def __init__(self, config: ManifoldConfig):
        self.config = config
        self.device: str = _default_device()

        # Default carrier sim parameters (not exposed as knobs).
        self._grid_size = (64, 64, 64)
        self._dt = 0.01

        # Tokenizer parameters (collision control knob).
        self._hash_prime = int(getattr(self.config, "hash_prime", 31))
        self._hash_vocab_size = int(getattr(self.config, "hash_vocab_size", 4096))
        if self._hash_vocab_size <= 0 or (self._hash_vocab_size & (self._hash_vocab_size - 1)) != 0:
            raise ValueError("ManifoldConfig.hash_vocab_size must be a power-of-two (e.g., 4096, 2048, 1024, ...)")
        self._hash_mask = self._hash_vocab_size - 1

        # Dashboard requires spatial state.
        if self.config.dashboard and not bool(getattr(self.config, "enable_spatial_physics", True)):
            raise ValueError("ManifoldConfig.dashboard=True requires enable_spatial_physics=True")

        self.physics = (
            ManifoldPhysics(
                config=ManifoldPhysicsConfig(
                    grid_size=self._grid_size,
                    dt=self._dt,
                    device=self.device,
                ),
            )
            if bool(getattr(self.config, "enable_spatial_physics", True))
            else None
        )

        self.carriers = SpectralCarrierPhysics(
            config=SpectralCarrierConfig(),
            grid_size=self._grid_size,
            dt=self._dt,
            device=self.device,
        )

        self._token_ids: Optional[torch.Tensor] = None
        self._osc_phase: Optional[torch.Tensor] = None
        self._osc_omega: Optional[torch.Tensor] = None
        self._osc_energy: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None
        self._vel: Optional[torch.Tensor] = None
        self._mass: Optional[torch.Tensor] = None
        self._heat: Optional[torch.Tensor] = None
        self._carrier_out: Optional[dict[str, torch.Tensor]] = None
        self._meta: Optional[List[Optional[Any]]] = None

        self._dashboard: Optional[SimulationDashboard] = None
        self._halt_mass: float = 0.0  # For confidence-based halting
        self._viz_idx: Optional[torch.Tensor] = None  # Sampled indices for visualization
        self._viz_n: Optional[int] = None  # Track when to resample visualization indices

    def _log(self, msg: str) -> None:
        if bool(getattr(self.config, "verbose", True)):
            print(msg)

    def load(self) -> None:
        """Load bytes and build oscillator tensors."""
        # Get data as a flat numpy uint8 array.
        meta_per_byte: Optional[List[Optional[Any]]] = None
        if self.config.bytes_data is not None:
            data = self.config.bytes_data
            if isinstance(data, np.ndarray):
                if data.dtype != np.uint8:
                    data = data.astype(np.uint8, copy=False)
            else:
                data = np.frombuffer(data, dtype=np.uint8)

        elif self.config.generator is not None:
            buf = bytearray()
            meta_list: List[Optional[Any]] = []
            all_single_byte = True
            for meta, payload in PrefetchIterator(_iter_chunks(self.config.generator)):
                if len(payload) != 1:
                    all_single_byte = False
                if all_single_byte:
                    meta_list.append(meta)
                buf.extend(payload)
            data = np.frombuffer(buf, dtype=np.uint8)
            meta_per_byte = meta_list if all_single_byte and len(meta_list) == len(data) else None

        else:
            raise ValueError("ManifoldConfig needs 'generator' or 'bytes_data'")

        n = len(data)
        if n == 0:
            self._token_ids = torch.empty(0, device=self.device, dtype=torch.int64)
            self._osc_phase = torch.empty(0, device=self.device, dtype=torch.float32)
            self._osc_omega = torch.empty(0, device=self.device, dtype=torch.float32)
            self._osc_energy = torch.empty(0, device=self.device, dtype=torch.float32)
            self._carrier_out = None
            self._meta = meta_per_byte
            return

        # To torch (device-first to avoid extra CPU-side transforms).
        byte_t = torch.from_numpy(data).to(device=self.device, dtype=torch.uint8)
        p = torch.arange(n, device=self.device, dtype=torch.int64)

        # Positions (reset every segment_size if set)
        seg = self.config.segment_size
        pos = p if not seg else torch.remainder(p, int(seg))

        # Hash: (byte * prime + pos) % vocab (bitmask fast-path; vocab must be power-of-two).
        tid = (byte_t.to(torch.int64) * self._hash_prime + pos) & self._hash_mask

        if bool(getattr(self.config, "store_token_ids", True)):
            self._token_ids = tid
        else:
            self._token_ids = None

        # Optional spatial initialization (for dashboards / spatial physics).
        if self.physics is not None:
            # Positions from token hash (initial state) - keep as integer math for stability.
            gx, gy, gz = self._grid_size
            self._pos = torch.stack(
                [
                    (tid * 73856093) % int(gx),
                    (tid * 19349663) % int(gy),
                    (tid * 83492791) % int(gz),
                ],
                dim=1,
            ).to(dtype=torch.float32)

            self._vel = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
            self._mass = torch.ones(n, device=self.device, dtype=torch.float32)
            self._heat = torch.zeros(n, device=self.device, dtype=torch.float32)
        else:
            self._pos = None
            self._vel = None
            self._mass = None
            self._heat = None

        tid_f = tid.to(torch.float32)
        self._osc_omega = tid_f * (2.0 / float(self._hash_vocab_size))
        two_pi = float(2.0 * math.pi)
        self._osc_phase = torch.remainder(tid_f * 0.0001 + p.to(torch.float32) * 0.01, two_pi)
        self._osc_energy = torch.ones(n, device=self.device, dtype=torch.float32)
        self._carrier_out = None
        self._meta = meta_per_byte

    def confidence(self, out: dict[str, torch.Tensor]) -> float:
        """Compute confidence from carrier state.
        
        Confidence = (crystallized_amplitude / total_amplitude) * (1 - mean_conflict)
        
        High confidence when:
        - Most amplitude is in crystallized carriers (stable features)
        - Low conflict (coherent coupling)
        """
        num_carriers = int(getattr(self.carriers, "num_carriers", 0))
        if num_carriers == 0:
            return 0.0

        amp = out.get("amplitudes")
        cs = out.get("carrier_state")
        conf = out.get("conflict")

        if amp is None or cs is None:
            return 0.0

        amp_slice = amp[:num_carriers].to(torch.float32)
        cs_slice = cs[:num_carriers]
        eps = 1e-8
        total_amp = amp_slice.sum()
        crystallized_amp = (amp_slice * (cs_slice == 2).to(torch.float32)).sum()
        amp_ratio = crystallized_amp / (total_amp + eps)
        mean_conflict = conf[:num_carriers].to(torch.float32).mean() if conf is not None else torch.tensor(0.0, device=amp_slice.device)
        score = amp_ratio * (1.0 - torch.clamp(mean_conflict, 0.0, 1.0))
        return float(score.item())

    def _is_ready(self, out: dict[str, torch.Tensor], step: int) -> bool:
        ready = self.config.ready
        if callable(ready):
            return bool(ready(self, out, int(step)))

        mode = str(ready).strip().lower()

        if mode in ("confidence",):
            # Accumulate halt_mass based on confidence (like old system)
            conf = self.confidence(out)
            dt = self._dt
            eps = 1e-8
            self._halt_mass = min(1.0, self._halt_mass + (conf * dt) / (conf + eps))
            return self._halt_mass >= 1.0

        # Legacy modes
        num_carriers = int(getattr(self.carriers, "num_carriers", 0))
        cs = out.get("carrier_state")
        if cs is None or num_carriers <= 0:
            num_crystallized = 0
        else:
            num_crystallized = int((cs[:num_carriers] == 2).sum().item())

        if mode in ("crystallized", "crystalized"):
            return num_crystallized > 0

        if mode in ("thinking_complete", "thinking-complete", "converged", "convergence"):
            if num_carriers < self.config.ready_min_carriers:
                return False
            conf_tensor = out.get("conflict")
            if conf_tensor is not None and num_carriers > 0:
                mean_conflict = conf_tensor[:num_carriers].mean().item()
            else:
                mean_conflict = 0.0
            if mean_conflict > self.config.ready_max_conflict:
                return False
            if self.config.ready_min_crystallized_frac > 0.0:
                frac = num_crystallized / max(1, num_carriers)
                if frac < self.config.ready_min_crystallized_frac:
                    return False
            return True

        raise ValueError(f"Unknown ready condition: {ready!r}")

    def _dashboard_video_path(self) -> Path:
        if self.config.dashboard_video_path is not None:
            return Path(self.config.dashboard_video_path)

        env = os.environ.get("THERMO_MANIFOLD_DASHBOARD_VIDEO_PATH")
        if env:
            return Path(env)

        # Fallback: repo_root/paper/artifacts/video/<name>_<timestamp>.mp4
        name = os.environ.get("THERMO_MANIFOLD_EXPERIMENT_NAME", "experiment")
        ts = time.strftime("%Y%m%d_%H%M%S")
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "paper" / "artifacts" / "video" / f"{name}_{ts}.mp4"

    def _ensure_dashboard(self) -> None:
        if not self.config.dashboard or self._dashboard is not None:
            return
        self._dashboard = SimulationDashboard(
            grid_size=self._grid_size,
            device=self.device,
        )
        self._dashboard.start_recording(
            self._dashboard_video_path(),
            fps=self.config.dashboard_video_fps,
        )

    _MAX_VIZ_PARTICLES: int = 5000  # Sample for visualization

    def _dashboard_update(self, step: int, out: dict, step_time_ms: float) -> None:
        if self._dashboard is None:
            return

        # Dashboard requires spatial state.
        if self._pos is None or self._vel is None or self._heat is None:
            return

        n = int(self._pos.shape[0])
        if n <= 0:
            return

        if n > self._MAX_VIZ_PARTICLES:
            # Sample indices once per size; randint avoids O(n) randperm.
            if self._viz_idx is None or self._viz_n != n or int(self._viz_idx.numel()) != self._MAX_VIZ_PARTICLES:
                self._viz_idx = torch.randint(0, n, (self._MAX_VIZ_PARTICLES,), device=self.device)
                self._viz_n = n
            idx = self._viz_idx
            energy = self._osc_energy[idx]
            omega = self._osc_omega[idx]
            pos = self._pos[idx]
            heat = self._heat[idx]
            vel = self._vel[idx]
        else:
            energy = self._osc_energy
            omega = self._osc_omega
            pos = self._pos
            heat = self._heat
            vel = self._vel

        self._dashboard.update(
            step=step,
            positions=pos,
            velocities=vel,
            energies=energy,
            heats=heat,
            excitations=omega,
            masses=(self._mass[idx] if self._viz_enabled else self._mass),
            step_time_ms=step_time_ms,
            carriers=CarrierState(
                frequencies=out.get("frequencies", torch.empty(0, device=self.device)),
                gate_widths=out.get("gate_widths", torch.empty(0, device=self.device)),
                amplitudes=out.get("amplitudes", torch.empty(0, device=self.device)),
                phases=out.get("phases", torch.empty(0, device=self.device)),
            ),
        )

    def run(self) -> dict:
        """Run carrier dynamics until ready, then return the final state snapshot."""
        self._log("[manifold] run() starting...")
        
        if self._osc_phase is None:
            self._log("[manifold] loading data...")
            t0 = time.perf_counter()
            self.load()
            self._log(
                f"[manifold] load complete: {int(self._osc_phase.numel() if self._osc_phase is not None else 0):,} "
                f"oscillators in {time.perf_counter()-t0:.2f}s"
            )
        
        assert self._osc_phase is not None
        assert self._osc_omega is not None
        assert self._osc_energy is not None

        n_osc = self._osc_phase.numel()
        self._log(f"[manifold] oscillators: {n_osc:,}")

        if n_osc == 0:
            self._log("[manifold] no oscillators, returning empty state")
            self._carrier_out = {
                "frequencies": torch.empty(0, device=self.device, dtype=torch.float32),
                "gate_widths": torch.empty(0, device=self.device, dtype=torch.float32),
                "amplitudes": torch.empty(0, device=self.device, dtype=torch.float32),
                "phases": torch.empty(0, device=self.device, dtype=torch.float32),
                "conflict": torch.empty(0, device=self.device, dtype=torch.float32),
                "osc_phase": self._osc_phase,
                "osc_energy": self._osc_energy,
                "carrier_state": torch.empty(0, device=self.device, dtype=torch.int32),
                "carrier_age": torch.empty(0, device=self.device, dtype=torch.int32),
            }
            return self.state

        interval = int(self.config.dashboard_update_interval)
        max_steps = int(self.config.max_steps)
        self._log(f"[manifold] max_steps={max_steps}, dashboard_interval={interval}, ready={self.config.ready}")

        self._log("[manifold] ensuring dashboard...")
        self._ensure_dashboard()
        self._log(f"[manifold] dashboard: {'active' if self._dashboard else 'disabled'}")

        self._log("[manifold] starting simulation loop...")
        try:
            for step in range(max_steps):
                t0 = time.perf_counter()
                
                # Spatial physics step
                if self.physics is not None:
                    self._pos, self._vel, self._osc_energy, self._heat, self._osc_omega = self.physics.step(
                        self._pos, self._vel, self._osc_energy, self._heat, self._osc_omega, self._mass
                    )

                # Spectral step
                out = self.carriers.step(self._osc_phase, self._osc_omega, self._osc_energy)
                step_ms = (time.perf_counter() - t0) * 1000.0

                self._osc_phase = out["osc_phase"]
                self._osc_energy = out.get("osc_energy", self._osc_energy)
                self._carrier_out = out

                # Log every N steps or first few.
                log_every = max(1, int(getattr(self.config, "log_every", 10)))
                if step < 5 or step % log_every == 0:
                    n_carriers = int(getattr(self.carriers, "num_carriers", 0))
                    conf = self.confidence(out)
                    self._log(
                        f"[manifold] step {step}: {step_ms:.1f}ms, carriers={n_carriers}, conf={conf:.3f}, halt={self._halt_mass:.3f}"
                    )

                # OPTIMIZATION: Only update dashboard every N steps (respects dashboard_update_interval)
                if self._dashboard is not None and (step % interval == 0 or step < 5):
                    t1 = time.perf_counter()
                    self._dashboard_update(step=step, out=out, step_time_ms=step_ms)
                    dash_ms = (time.perf_counter() - t1) * 1000.0
                    if step < 5 or step % 50 == 0:
                        self._log(f"[manifold] dashboard update: {dash_ms:.1f}ms")

                if self._is_ready(out, step):
                    self._log(f"[manifold] ready at step {step}, returning")
                    return self.state

            self._log(f"[manifold] reached max_steps={max_steps}")

        finally:
            if self._dashboard is not None:
                self._log("[manifold] stopping dashboard...")
                try:
                    self._dashboard.stop_recording()
                except Exception:
                    pass
                try:
                    self._dashboard.close()
                except Exception:
                    pass
                self._dashboard = None

        return self.state

    @property
    def state(self) -> dict:
        """Final state snapshot for observers."""
        carrier_out = self._carrier_out or {}
        cs = carrier_out.get("carrier_state")
        n_carriers = int(getattr(self.carriers, "num_carriers", 0))
        if (n_carriers <= 0) and cs is not None:
            n_carriers = int(cs.numel())
        n_crystallized = (
            int((cs[:n_carriers] == 2).sum().item())
            if cs is not None and cs.numel() > 0 and n_carriers > 0
            else 0
        )
        return {
            "token_ids": self._token_ids,
            "osc_phase": self._osc_phase,
            "osc_omega": self._osc_omega,
            "osc_energy": self._osc_energy,
            "meta": self._meta,
            "carriers": carrier_out,
            "num_carriers": n_carriers,
            "num_crystallized": n_crystallized,
        }

