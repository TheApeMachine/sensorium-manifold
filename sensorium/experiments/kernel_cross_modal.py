"""Cross-modal experiment on the real manifold runner.

This version is strict:
- uses `run_stream_on_manifold` (MPS only for publication rows)
- forbids analysis fallback
- records per-run provenance
- uses only actual manifold state for visual/metric outputs
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sensorium.dataset import (
    CrossModalConfig,
    CrossModalDataset,
    create_checkerboard_image,
    create_stripe_image,
)
from sensorium.experiments.base import Experiment
from sensorium.experiments.manifold_runner import ManifoldRunConfig, run_stream_on_manifold
from sensorium.observers.inference import InferenceObserver
from sensorium.projectors import ConsoleProjector, PipelineProjector
from sensorium.projectors.cross_modal import (
    CrossModalFigureConfig,
    CrossModalFigureProjector,
    CrossModalTableProjector,
)


class KernelCrossModal(Experiment):
    """Cross-modal text+image evidence using the current manifold stack."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
    ):
        reportable = [
            "scenario",
            "seed",
            "n_particles",
            "n_image_bytes",
            "n_text_bytes",
            "mse",
            "mae",
            "psnr",
            "image_mass_share",
            "text_mass_share",
            "image_energy_mean",
            "text_energy_mean",
            "image_text_centroid_dist",
            "text_label_coverage",
            "run_backend",
            "run_steps",
            "run_termination",
            "init_ms",
            "simulate_ms",
        ]
        super().__init__(
            experiment_name,
            profile,
            dashboard=dashboard,
            reportable=reportable,
        )
        self.seeds = tuple(self.experiment_seeds(default=(7, 19, 43)))
        self.run_config = ManifoldRunConfig(
            grid_size=(64, 64, 64),
            max_steps=220,
            min_steps=50,
            allow_analysis_fallback=False,
        )
        self.dataset_config = CrossModalConfig(top_k_freq=96, image_size=32)
        self.scenarios = (
            {
                "name": "horizontal",
                "image": create_stripe_image(32, "horizontal"),
                "labels": ["horizontal", "stripes", "lines"],
            },
            {
                "name": "vertical",
                "image": create_stripe_image(32, "vertical"),
                "labels": ["vertical", "stripes", "lines"],
            },
            {
                "name": "diagonal",
                "image": create_stripe_image(32, "diagonal"),
                "labels": ["diagonal", "stripes", "pattern"],
            },
            {
                "name": "checkerboard",
                "image": create_checkerboard_image(32),
                "labels": ["checkerboard", "grid", "pattern"],
            },
        )

        self._run_rows: list[dict[str, Any]] = []
        self._dashboard_tags: set[str] = set()
        self._provenance_path: str = ""

        self.inference = InferenceObserver([])
        self.projector = PipelineProjector(
            ConsoleProjector(fields=reportable, format="table"),
            CrossModalFigureProjector(
                config=CrossModalFigureConfig(name="cross_modal"),
                output_dir=Path("paper/figures"),
            ),
            CrossModalTableProjector(output_dir=Path("paper/tables")),
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
    def _to_numpy_f64(x: Any, *, default: float = 0.0, length: int | None = None) -> np.ndarray:
        if x is None:
            if isinstance(length, int) and length > 0:
                return np.full((length,), float(default), dtype=np.float64)
            return np.asarray([], dtype=np.float64)
        if hasattr(x, "detach"):
            t = x.detach().to("cpu")
            if hasattr(t, "to"):
                t = t.to(dtype=torch.float32)
            return np.asarray(t.numpy(), dtype=np.float64)
        arr2 = np.asarray(x, dtype=np.float64)
        if isinstance(length, int) and length > 0 and arr2.size == 0:
            return np.full((length,), float(default), dtype=np.float64)
        return arr2

    @staticmethod
    def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        if points.size == 0 or weights.size == 0:
            return None
        wsum = float(np.sum(weights))
        if wsum <= 0.0:
            return None
        return np.sum(points * weights[:, None], axis=0) / wsum

    @staticmethod
    def _label_spans(labels: list[str], *, image_prefix: int) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for idx, label in enumerate(labels):
            b = label.encode("utf-8")
            start = int(image_prefix + cursor)
            end = int(start + len(b))
            spans.append((label, start, end))
            cursor += int(len(b))
            if idx < len(labels) - 1:
                cursor += 1
        return spans

    @staticmethod
    def _reconstruct_image(
        dataset: CrossModalDataset,
        seq: np.ndarray,
        masses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        metadata = dataset.metadata
        topk_indices = np.asarray(metadata.get("topk_indices", []), dtype=np.int64)
        mag_selected = np.asarray(metadata.get("mag_selected", []), dtype=np.float64)
        phase_selected = np.asarray(metadata.get("phase_selected", []), dtype=np.float64)
        shape = tuple(int(x) for x in metadata.get("shape", (32, 32)))
        if topk_indices.size == 0:
            z = np.zeros(shape, dtype=np.float32)
            return z, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        n_img = int(len(dataset.image_bytes))
        n_total = int(max(int(np.max(seq)) + 1 if seq.size else 0, n_img))
        mass_by_seq = np.bincount(seq, weights=masses, minlength=n_total).astype(np.float64)

        # Derive per-frequency weights from paired magnitude/phase byte masses.
        weights: list[float] = []
        for i in range(int(topk_indices.size)):
            bi = int(2 * i)
            bj = int(2 * i + 1)
            if bi >= n_img:
                break
            w_mag = float(mass_by_seq[bi]) if bi < mass_by_seq.size else 0.0
            w_phase = float(mass_by_seq[bj]) if bj < mass_by_seq.size else 0.0
            weights.append(0.5 * (w_mag + w_phase))
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.size == 0:
            z = np.zeros(shape, dtype=np.float32)
            return z, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        ref = float(np.median(w_arr[w_arr > 0])) if np.any(w_arr > 0) else 1.0
        if not math.isfinite(ref) or ref <= 0.0:
            ref = 1.0
        w_norm = np.clip(w_arr / ref, 0.10, 4.00)

        h, w = shape
        spectrum = np.zeros((h, w), dtype=np.complex64)
        n_coeff = int(min(topk_indices.size, mag_selected.size, phase_selected.size, w_norm.size))
        for i in range(n_coeff):
            idx = int(topk_indices[i])
            u = int(idx // w)
            v = int(idx % w)
            if not (0 <= u < h and 0 <= v < w):
                continue
            mag = float(mag_selected[i]) * float(w_norm[i])
            phase = float(phase_selected[i])
            spectrum[u, v] = np.complex64(mag * np.exp(1j * phase))

        recon = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
        rmin = float(np.min(recon))
        rmax = float(np.max(recon))
        if rmax > rmin:
            recon = (recon - rmin) / (rmax - rmin)
        else:
            recon = np.clip(recon, 0.0, 1.0)

        # Frequency plot vectors.
        u_all = (topk_indices[:n_coeff] // w) - (h // 2)
        v_all = (topk_indices[:n_coeff] % w) - (w // 2)
        energy_all = np.asarray(mag_selected[:n_coeff], dtype=np.float64) * np.asarray(
            w_norm[:n_coeff], dtype=np.float64
        )

        return recon.astype(np.float32), u_all.astype(np.float64), v_all.astype(np.float64), energy_all.astype(np.float64)

    def _analyze_run(
        self,
        *,
        dataset: CrossModalDataset,
        state: dict,
    ) -> dict[str, Any]:
        seq = self._to_numpy_i64(state.get("sequence_indices"))
        positions = self._to_numpy_f64(state.get("positions"))
        if positions.ndim == 1 and positions.size > 0:
            positions = np.reshape(positions, (-1, 3))
        masses = self._to_numpy_f64(state.get("masses"), default=1.0, length=int(seq.size))
        energies = self._to_numpy_f64(state.get("energies"), default=0.0, length=int(seq.size))

        n_particles = int(seq.size)
        n_image_bytes = int(len(dataset.image_bytes))
        n_text_bytes = int(len(dataset.text_bytes))
        n_total_bytes = int(n_image_bytes + n_text_bytes)

        image_mask = (seq >= 0) & (seq < n_image_bytes)
        text_mask = (seq >= n_image_bytes) & (seq < n_total_bytes)

        total_mass = float(np.sum(masses)) if masses.size else 0.0
        image_mass = float(np.sum(masses[image_mask])) if masses.size else 0.0
        text_mass = float(np.sum(masses[text_mask])) if masses.size else 0.0
        image_mass_share = float(image_mass / total_mass) if total_mass > 0.0 else 0.0
        text_mass_share = float(text_mass / total_mass) if total_mass > 0.0 else 0.0

        image_energy_mean = float(np.mean(energies[image_mask])) if np.any(image_mask) else 0.0
        text_energy_mean = float(np.mean(energies[text_mask])) if np.any(text_mask) else 0.0

        image_centroid = self._weighted_centroid(positions[image_mask], masses[image_mask]) if positions.size else None

        label_centroids: list[dict[str, Any]] = []
        dists: list[float] = []
        spans = self._label_spans(dataset.text_labels, image_prefix=n_image_bytes)
        for label, s0, s1 in spans:
            mask = (seq >= int(s0)) & (seq < int(s1))
            if not np.any(mask):
                continue
            centroid = self._weighted_centroid(positions[mask], masses[mask])
            if centroid is None:
                continue
            mass = float(np.sum(masses[mask]))
            label_centroids.append(
                {
                    "label": str(label),
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]),
                    "mass": mass,
                }
            )
            if image_centroid is not None:
                dists.append(float(np.linalg.norm(centroid - image_centroid)))

        text_label_coverage = float(len(label_centroids) / max(1, len(dataset.text_labels)))
        image_text_centroid_dist = float(np.mean(dists)) if dists else 0.0

        recon, fu, fv, fe = self._reconstruct_image(dataset, seq, masses)
        original = np.asarray(dataset.image, dtype=np.float32)
        mse = float(np.mean((original - recon) ** 2))
        mae = float(np.mean(np.abs(original - recon)))
        psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12))) if mse > 0.0 else 100.0

        vis = {
            "original": original,
            "reconstructed": recon,
            "freq_u": fu,
            "freq_v": fv,
            "freq_energy": fe,
            "image_points": positions[image_mask] if positions.size else np.zeros((0, 3), dtype=np.float64),
            "image_point_energy": energies[image_mask] if energies.size else np.asarray([], dtype=np.float64),
            "text_points": positions[text_mask] if positions.size else np.zeros((0, 3), dtype=np.float64),
            "text_point_energy": energies[text_mask] if energies.size else np.asarray([], dtype=np.float64),
            "text_label_centroids": label_centroids,
            "text_labels": [str(x) for x in dataset.text_labels],
        }

        return {
            "n_particles": n_particles,
            "n_image_bytes": n_image_bytes,
            "n_text_bytes": n_text_bytes,
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "image_mass_share": image_mass_share,
            "text_mass_share": text_mass_share,
            "image_energy_mean": image_energy_mean,
            "text_energy_mean": text_energy_mean,
            "image_text_centroid_dist": image_text_centroid_dist,
            "text_label_coverage": text_label_coverage,
            "visual": vis,
        }

    def run(self):
        self._run_rows = []
        self._dashboard_tags = set()
        self._provenance_path = ""

        for item in self.scenarios:
            name = str(item["name"])
            image = np.asarray(item["image"], dtype=np.float32)
            labels = [str(x) for x in item["labels"]]

            for seed in self.seeds:
                dataset = CrossModalDataset(
                    image=image,
                    text_labels=labels,
                    config=self.dataset_config,
                )
                run_name = f"{self.experiment_name}_{name}_s{int(seed)}"
                record_dashboard = bool(self.dashboard) and name not in self._dashboard_tags
                on_step = None
                if record_dashboard:
                    self.start_dashboard(
                        grid_size=self.run_config.grid_size,
                        run_name=run_name,
                    )

                    def _on_step(state: dict) -> None:
                        self.dashboard_update(state)

                    on_step = _on_step

                try:
                    state, meta = run_stream_on_manifold(
                        dataset.generate(),
                        config=self.run_config,
                        on_step=on_step,
                    )
                finally:
                    if record_dashboard:
                        self.close_dashboard()
                        self._dashboard_tags.add(name)

                termination = str(meta.get("run_termination", ""))
                if termination != "quiet":
                    raise RuntimeError(
                        f"[cross_modal] expected quiet termination but got '{termination}' "
                        f"(scenario={name}, seed={seed}, max_steps={self.run_config.max_steps})"
                    )

                analysis = self._analyze_run(dataset=dataset, state=state)

                scalar_row = {
                    "scenario": name,
                    "seed": int(seed),
                    "run_name": run_name,
                    "run_backend": str(meta.get("run_backend", "")),
                    "run_steps": int(meta.get("run_steps", 0) or 0),
                    "run_termination": termination,
                    "init_ms": float(meta.get("init_ms", 0.0) or 0.0),
                    "simulate_ms": float(meta.get("simulate_ms", 0.0) or 0.0),
                    "n_particles": int(analysis["n_particles"]),
                    "n_image_bytes": int(analysis["n_image_bytes"]),
                    "n_text_bytes": int(analysis["n_text_bytes"]),
                    "mse": float(analysis["mse"]),
                    "mae": float(analysis["mae"]),
                    "psnr": float(analysis["psnr"]),
                    "image_mass_share": float(analysis["image_mass_share"]),
                    "text_mass_share": float(analysis["text_mass_share"]),
                    "image_energy_mean": float(analysis["image_energy_mean"]),
                    "text_energy_mean": float(analysis["text_energy_mean"]),
                    "image_text_centroid_dist": float(analysis["image_text_centroid_dist"]),
                    "text_label_coverage": float(analysis["text_label_coverage"]),
                }
                self._run_rows.append(dict(scalar_row))

                # Keep visual payload for figure projectors.
                row_with_visuals = dict(scalar_row)
                row_with_visuals.update(analysis["visual"])
                self.inference.observe({}, **row_with_visuals)

        self.assert_allowed_backends(self._run_rows, allowed=("mps",))
        self._provenance_path = str(
            self.write_provenance_jsonl(self._run_rows, stem=f"{self.experiment_name}_raw")
        )
        for row in self.inference.results:
            row["provenance_jsonl"] = self._provenance_path

        return self.project()

    def observe(self, state: dict) -> dict:
        return {}

    def project(self) -> dict:
        outputs = self.projector.project(self.inference)
        return {
            "projectors": outputs,
            "provenance_jsonl": self._provenance_path,
        }
