"""MNIST inpainting experiment on the real manifold runner.

Strict experiment policy:
- uses `run_stream_on_manifold` (real MPS manifold only)
- forbids analysis fallback
- requires quiet-settled termination for claim-bearing runs
- writes per-seed provenance and paper artifacts
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sensorium.dataset import MNISTConfig, MNISTDataset, MNIST_IMAGE_SIZE
from sensorium.experiments.base import Experiment
from sensorium.experiments.manifold_runner import ManifoldRunConfig, run_stream_on_manifold
from sensorium.observers.inference import InferenceObserver
from sensorium.projectors import ConsoleProjector, PipelineProjector
from sensorium.projectors.image import ImageFigureConfig, ImageFigureProjector, ImageTableProjector


class KernelImageGen(Experiment):
    """MNIST inpainting via real manifold state statistics."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
    ):
        reportable = [
            "seed",
            "n_particles",
            "run_backend",
            "run_steps",
            "run_termination",
            "init_ms",
            "simulate_ms",
            "psnr_10",
            "psnr_20",
            "psnr_30",
            "psnr_50",
            "mae_10",
            "mae_20",
            "mae_30",
            "mae_50",
        ]
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=reportable,
        )

        self.train_images = 100
        self.test_images = 20
        self.mask_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
        self.seeds = tuple(self.experiment_seeds(default=(7, 19, 43)))
        self.run_config = ManifoldRunConfig(
            grid_size=(64, 64, 64),
            max_steps=220,
            min_steps=50,
            allow_analysis_fallback=False,
        )
        self.settle_max_steps_cap = 4096
        self.settle_growth = 2

        self._run_rows: list[dict[str, Any]] = []
        self._provenance_path: str = ""

        self.inference = InferenceObserver([])
        self.projector = PipelineProjector(
            ConsoleProjector(fields=reportable, format="table"),
            ImageTableProjector(
                output_dir=Path("paper/tables"),
                train_images=int(self.train_images),
                test_images=int(self.test_images),
            ),
            ImageFigureProjector(
                config=ImageFigureConfig(name="image_gen"),
                output_dir=Path("paper/figures"),
            ),
        )

    @staticmethod
    def _to_numpy_i64(x: Any) -> np.ndarray:
        if x is None:
            return np.asarray([], dtype=np.int64)
        if hasattr(x, "detach"):
            t = x.detach().to("cpu")
            if hasattr(t, "to"):
                t = t.to(dtype=torch.int64)
            return np.asarray(t.numpy(), dtype=np.int64)
        return np.asarray(x, dtype=np.int64)

    @staticmethod
    def _to_numpy_f64(x: Any, *, length: int | None = None, default: float = 0.0) -> np.ndarray:
        if x is None:
            if isinstance(length, int) and length > 0:
                return np.full((length,), float(default), dtype=np.float64)
            return np.asarray([], dtype=np.float64)
        if hasattr(x, "detach"):
            t = x.detach().to("cpu")
            if hasattr(t, "to"):
                t = t.to(dtype=torch.float32)
            return np.asarray(t.numpy(), dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        if isinstance(length, int) and length > 0 and arr.size == 0:
            return np.full((length,), float(default), dtype=np.float64)
        return arr

    @staticmethod
    def _neighbors(pos: int) -> list[int]:
        w = 28
        row = int(pos // w)
        col = int(pos % w)
        out: list[int] = []
        if row > 0:
            out.append((row - 1) * w + col)
        if row < w - 1:
            out.append((row + 1) * w + col)
        if col > 0:
            out.append(row * w + (col - 1))
        if col < w - 1:
            out.append(row * w + (col + 1))
        return out

    def _build_position_model(self, state: dict) -> np.ndarray:
        seq = self._to_numpy_i64(state.get("sequence_indices"))
        byt = self._to_numpy_i64(state.get("byte_values"))
        masses = self._to_numpy_f64(state.get("masses"), length=int(seq.size), default=1.0)

        model = np.zeros((MNIST_IMAGE_SIZE, 256), dtype=np.float64)
        if seq.size == 0 or byt.size == 0:
            model += 1.0 / 256.0
            return model

        valid = (
            (seq >= 0)
            & (seq < int(MNIST_IMAGE_SIZE))
            & (byt >= 0)
            & (byt < 256)
        )
        if np.any(valid):
            np.add.at(model, (seq[valid], byt[valid]), masses[valid])

        # Small smoothing to avoid zero-probability collapse.
        model += 1e-6
        denom = np.sum(model, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return model / denom

    def _neighbor_distribution(
        self,
        image: np.ndarray,
        pos: int,
        masked: set[int],
        *,
        sigma: float = 24.0,
    ) -> np.ndarray:
        vals: list[int] = []
        for npos in self._neighbors(int(pos)):
            if int(npos) in masked:
                continue
            vals.append(int(image[int(npos)]))
        if not vals:
            return np.full((256,), 1.0 / 256.0, dtype=np.float64)

        x = np.arange(256, dtype=np.float64)
        score = np.zeros((256,), dtype=np.float64)
        inv = 1.0 / (2.0 * sigma * sigma)
        for v in vals:
            d = x - float(v)
            score += np.exp(-(d * d) * inv)
        s = float(np.sum(score))
        if s <= 0.0:
            return np.full((256,), 1.0 / 256.0, dtype=np.float64)
        return score / s

    def _inpaint(
        self,
        corrupted: bytes,
        mask_positions: list[int],
        pos_model: np.ndarray,
    ) -> bytes:
        img = np.frombuffer(corrupted, dtype=np.uint8).copy()
        masked: set[int] = {int(p) for p in mask_positions}
        order = [int(p) for p in mask_positions]

        # Two sweeps so newly predicted neighbors can support later pixels.
        for _ in range(2):
            for pos in order:
                prior = pos_model[int(pos)]
                neigh = self._neighbor_distribution(img, int(pos), masked)
                score = (0.75 * prior) + (0.25 * neigh)
                img[int(pos)] = np.uint8(int(np.argmax(score)))
                if int(pos) in masked:
                    masked.remove(int(pos))

        return bytes(img.tolist())

    @staticmethod
    def _metrics(original: bytes, reconstructed: bytes) -> dict[str, float]:
        a = np.frombuffer(original, dtype=np.uint8).astype(np.float64)
        b = np.frombuffer(reconstructed, dtype=np.uint8).astype(np.float64)
        mae = float(np.mean(np.abs(a - b)))
        mse = float(np.mean((a - b) ** 2))
        psnr = float(10.0 * math.log10((255.0 * 255.0) / max(mse, 1e-12)))
        return {"mae": mae, "mse": mse, "psnr": psnr}

    def _evaluate_batch(
        self,
        *,
        pos_model: np.ndarray,
        test_images: list[bytes],
        test_labels: list[int],
        seed: int,
    ) -> tuple[dict[float, dict[str, float]], list[dict[str, Any]]]:
        rng = np.random.default_rng(int(seed))
        mask_results: dict[float, dict[str, float]] = {}
        examples: list[dict[str, Any]] = []

        for frac in self.mask_fracs:
            maes: list[float] = []
            mses: list[float] = []
            psnrs: list[float] = []
            for idx, img in enumerate(test_images):
                n_mask = int(round(float(frac) * float(MNIST_IMAGE_SIZE)))
                n_mask = max(1, min(int(MNIST_IMAGE_SIZE), int(n_mask)))
                mask_positions = rng.choice(
                    int(MNIST_IMAGE_SIZE),
                    size=int(n_mask),
                    replace=False,
                )
                mask_positions_list = [int(x) for x in mask_positions.tolist()]

                masked = bytearray(img)
                for pos in mask_positions_list:
                    masked[int(pos)] = 128
                reconstructed = self._inpaint(bytes(masked), mask_positions_list, pos_model)
                m = self._metrics(img, reconstructed)
                maes.append(float(m["mae"]))
                mses.append(float(m["mse"]))
                psnrs.append(float(m["psnr"]))

                # Save a few canonical visual examples at 30% mask.
                if abs(float(frac) - 0.3) < 1e-9 and len(examples) < 4:
                    examples.append(
                        {
                            "original": bytes(img),
                            "masked": bytes(masked),
                            "reconstructed": bytes(reconstructed),
                            "mask_frac": float(frac),
                            "mae": float(m["mae"]),
                            "psnr": float(m["psnr"]),
                            "label": int(test_labels[idx]) if idx < len(test_labels) else -1,
                        }
                    )

            mask_results[float(frac)] = {
                "mae": float(np.mean(maes)) if maes else 0.0,
                "mse": float(np.mean(mses)) if mses else 0.0,
                "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
                "n_images": float(len(test_images)),
            }

        return mask_results, examples

    @staticmethod
    def _frac_key(frac: float) -> str:
        return str(int(round(float(frac) * 100.0)))

    def run(self):
        self._run_rows = []
        self._provenance_path = ""

        data_dir = self.repo_root / "data" / "mnist"
        train_dataset = MNISTDataset(
            MNISTConfig(
                data_dir=data_dir,
                train=True,
                limit=int(self.train_images),
            )
        )
        test_dataset = MNISTDataset(
            MNISTConfig(
                data_dir=data_dir,
                train=False,
                limit=int(self.test_images),
            )
        )

        test_images = list(test_dataset.images)
        test_labels = [int(x) for x in test_dataset.labels]

        for idx_seed, seed in enumerate(self.seeds):
            run_name = f"{self.experiment_name}_s{int(seed)}"
            record_dashboard = bool(self.dashboard) and idx_seed == 0
            attempts = 0
            attempt_steps = int(max(1, int(self.run_config.max_steps)))
            state: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            while True:
                attempts += 1
                on_step = None
                if record_dashboard:
                    self.start_dashboard(
                        grid_size=self.run_config.grid_size, run_name=run_name
                    )

                    def _on_step(state: dict) -> None:
                        self.dashboard_update(state)

                    on_step = _on_step

                try:
                    attempt_cfg = replace(
                        self.run_config,
                        max_steps=int(attempt_steps),
                        min_steps=min(50, int(attempt_steps)),
                    )
                    state, meta = run_stream_on_manifold(
                        train_dataset.generate(),
                        config=attempt_cfg,
                        on_step=on_step,
                    )
                finally:
                    if record_dashboard:
                        self.close_dashboard()
                termination = str(meta.get("run_termination", ""))
                if (
                    termination != "quiet"
                    and int(attempt_steps) < int(self.settle_max_steps_cap)
                ):
                    attempt_steps = min(
                        int(self.settle_max_steps_cap),
                        int(
                            max(
                                int(attempt_steps) + 1,
                                int(attempt_steps) * int(self.settle_growth),
                            )
                        ),
                    )
                    continue
                break

            termination = str(meta.get("run_termination", ""))
            if termination != "quiet":
                raise RuntimeError(
                    f"[image_gen] expected quiet termination but got '{termination}' "
                    f"(seed={seed}, max_steps={attempt_steps}, attempts={attempts})"
                )

            pos_model = self._build_position_model(state)
            mask_results, examples = self._evaluate_batch(
                pos_model=pos_model,
                test_images=test_images,
                test_labels=test_labels,
                seed=int(seed),
            )

            row: dict[str, Any] = {
                "seed": int(seed),
                "run_name": run_name,
                "n_particles": int(meta.get("n_particles", 0) or 0),
                "run_backend": str(meta.get("run_backend", "")),
                "run_steps": int(meta.get("run_steps", 0) or 0),
                "run_termination": termination,
                "run_max_steps_budget": int(attempt_steps),
                "run_attempts": int(attempts),
                "init_ms": float(meta.get("init_ms", 0.0) or 0.0),
                "simulate_ms": float(meta.get("simulate_ms", 0.0) or 0.0),
                "train_images": int(self.train_images),
                "test_images": int(self.test_images),
                "mask_results": mask_results,
                "examples": examples,
            }

            for frac in self.mask_fracs:
                k = self._frac_key(frac)
                m = mask_results.get(float(frac), {})
                row[f"psnr_{k}"] = float(m.get("psnr", 0.0))
                row[f"mae_{k}"] = float(m.get("mae", 0.0))
                row[f"mse_{k}"] = float(m.get("mse", 0.0))

            self._run_rows.append(
                {
                    "seed": int(row["seed"]),
                    "run_name": str(row["run_name"]),
                    "n_particles": int(row["n_particles"]),
                    "run_backend": str(row["run_backend"]),
                    "run_steps": int(row["run_steps"]),
                    "run_termination": str(row["run_termination"]),
                    "init_ms": float(row["init_ms"]),
                    "simulate_ms": float(row["simulate_ms"]),
                    **{f"psnr_{self._frac_key(f)}": float(mask_results.get(float(f), {}).get("psnr", 0.0)) for f in self.mask_fracs},
                    **{f"mae_{self._frac_key(f)}": float(mask_results.get(float(f), {}).get("mae", 0.0)) for f in self.mask_fracs},
                    **{f"mse_{self._frac_key(f)}": float(mask_results.get(float(f), {}).get("mse", 0.0)) for f in self.mask_fracs},
                }
            )
            self.inference.observe({}, **row)

        self.assert_allowed_backends(self._run_rows, allowed=("mps",))
        self._provenance_path = str(
            self.write_provenance_jsonl(self._run_rows, stem=f"{self.experiment_name}_raw")
        )
        for row in self.inference.results:
            row["provenance_jsonl"] = self._provenance_path

        return self.project()

    def observe(self, state: dict):
        return {}

    def project(self) -> dict:
        outputs = self.projector.project(self.inference)
        return {
            "projectors": outputs,
            "provenance_jsonl": self._provenance_path,
        }
