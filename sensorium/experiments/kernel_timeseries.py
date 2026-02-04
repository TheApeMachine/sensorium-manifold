"""Kernel time-series forecasting (byte-quantized).

We quantize a synthetic time series to bytes and run next-byte prediction
using the same kernel engine as text.

NON-CHEATING DESIGN:
====================
This experiment uses a proper train/test split in time:
- Training: First portion of the time series
- Testing: Future portion (strictly after training period)
- No lookahead: Predictions only use past values

We test on synthetic signals with learnable patterns:
- Periodic signals (sine, square, sawtooth)
- Trend + seasonality (linear + periodic)
- Regime-switching (changes in pattern type)

Writes:
- `paper/tables/timeseries_summary.tex`
- `paper/figures/timeseries.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple

import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import (
    Manifold,
    SimulationConfig,
    GeometricSimulationConfig,
    SpectralSimulationConfig,
)
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.carrier import CarrierObserver


def quantize_to_bytes(values: np.ndarray, min_val: float = None, max_val: float = None) -> bytes:
    """Quantize float values to bytes (0-255).
    
    Args:
        values: Array of float values
        min_val: Minimum value for normalization (if None, use data min)
        max_val: Maximum value for normalization (if None, use data max)
    
    Returns:
        Byte string of quantized values
    """
    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()
    
    # Normalize to [0, 255]
    normalized = (values - min_val) / (max_val - min_val + 1e-10)
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    
    return bytes(quantized)


def dequantize_from_bytes(data: bytes, min_val: float, max_val: float) -> np.ndarray:
    """Dequantize bytes back to float values."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return arr / 255.0 * (max_val - min_val) + min_val


class SyntheticTimeSeries:
    """Generate synthetic time series with learnable patterns."""
    
    def __init__(
        self,
        length: int = 2000,
        series_type: str = "periodic",  # periodic, trend_seasonal, regime_switch
        seed: int = 42,
    ):
        self.length = length
        self.series_type = series_type
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        self.values = self._generate()
        self.min_val = self.values.min()
        self.max_val = self.values.max()
        self.bytes_data = quantize_to_bytes(self.values, self.min_val, self.max_val)
    
    def _generate(self) -> np.ndarray:
        """Generate the time series based on type."""
        t = np.arange(self.length, dtype=np.float32)
        
        if self.series_type == "periodic":
            # Multiple overlapping periodicities
            y = (
                50 * np.sin(2 * np.pi * t / 50) +    # Slow period
                20 * np.sin(2 * np.pi * t / 13) +    # Medium period
                10 * np.sin(2 * np.pi * t / 7) +     # Fast period
                5 * self._rng.randn(self.length)     # Noise
            )
        
        elif self.series_type == "trend_seasonal":
            # Linear trend + seasonal component
            trend = 0.02 * t
            seasonal = 30 * np.sin(2 * np.pi * t / 100)
            noise = 5 * self._rng.randn(self.length)
            y = trend + seasonal + noise
        
        elif self.series_type == "regime_switch":
            # Switch between two patterns at midpoint
            mid = self.length // 2
            y = np.zeros(self.length, dtype=np.float32)
            
            # Regime 1: Low-frequency sine
            y[:mid] = 50 * np.sin(2 * np.pi * t[:mid] / 100)
            
            # Regime 2: High-frequency sine (rule shift!)
            y[mid:] = 50 * np.sin(2 * np.pi * t[mid:] / 20)
            
            # Add noise
            y += 5 * self._rng.randn(self.length)
        
        elif self.series_type == "sawtooth":
            # Repeating sawtooth pattern
            period = 50
            y = (t % period) / period * 100 + 5 * self._rng.randn(self.length)
        
        else:
            raise ValueError(f"Unknown series type: {self.series_type}")
        
        return y
    
    def train_test_split(self, test_ratio: float = 0.2) -> Tuple[bytes, bytes]:
        """Split time series - training is BEFORE testing (no lookahead)."""
        split_idx = int(self.length * (1 - test_ratio))
        
        train_bytes = self.bytes_data[:split_idx]
        test_bytes = self.bytes_data[split_idx:]
        
        return train_bytes, test_bytes


class TimeSeriesPredictor:
    """Predict next time series value using dual-domain inference.
    
    For periodic signals, we exploit:
    - Geometric: particles at similar phases cluster together
    - Spectral: carriers couple oscillators with matching periodicity
    """
    
    def __init__(
        self, 
        vocab_size: int = 4096, 
        prime: int = 31,
        period_samples: int = 0,
    ):
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask = vocab_size - 1
        self.period_samples = period_samples
        
        # Will be set by learn_from_manifold
        self.inference = None
    
    def learn_from_manifold(self, geo_state: Dict, spec_state: Dict):
        """Learn from manifold state using dual-domain inference."""
        from sensorium.observers.dual_domain import DualDomainInference
        
        self.inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=spec_state,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
    
    def predict(self, context_indices: torch.Tensor, target_position: int) -> int:
        """Predict the next byte value using carrier coupling.
        
        For time series, we look at:
        1. Recent particles (geometric context)
        2. What carriers they couple to (spectral)
        3. What byte at the target position couples best to those carriers
        """
        if self.inference is None:
            return 128  # Default mid-value
        
        # Use dual-domain scoring
        scores = self.inference.score_candidate_bytes(
            context_indices=context_indices,
            target_position=target_position,
            segment_size=self.period_samples if self.period_samples > 0 else None,
        )
        
        return int(np.argmax(scores))


class KernelTimeSeries(Experiment):
    """Time-series forecasting experiment using byte-quantized values."""
    
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)
        
        self.length = 2000
        self.context_length = 10
        self.series_types = ["periodic", "trend_seasonal", "regime_switch", "sawtooth"]
        
        self.results: Dict[str, Dict[str, Any]] = {}

    def observe(self, state: dict):
        """Generate paper artifacts from all series results."""
        if not self.results:
            print("Warning: No results collected")
            return
        
        import matplotlib.pyplot as plt
        
        # Summary table
        summary = {}
        for series_type, res in self.results.items():
            summary[f"{series_type}_mae"] = res["mae"]
            summary[f"{series_type}_mse"] = res["mse"]
            summary[f"{series_type}_accuracy"] = res["accuracy"]
        
        summary["context_length"] = self.context_length
        summary["series_length"] = self.length
        
        self.write_kv_table("timeseries_summary", summary)
        
        # Multi-panel figure
        n_types = len(self.series_types)
        fig, axes = plt.subplots(n_types, 2, figsize=(14, 3 * n_types))
        
        for idx, series_type in enumerate(self.series_types):
            res = self.results.get(series_type, {})
            if not res:
                continue
            
            actuals = res.get("actuals", [])
            predictions = res.get("predictions", [])
            
            # Left: Actual vs Predicted
            ax = axes[idx, 0] if n_types > 1 else axes[0]
            ax.plot(actuals, label='Actual', color='#336699', linewidth=1.5)
            ax.plot(predictions, label='Predicted', color='#4C994C', 
                    linewidth=1.5, linestyle='--', alpha=0.8)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value (quantized)')
            ax.set_title(f'{series_type.replace("_", " ").title()}: Forecast vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Right: Error distribution
            ax = axes[idx, 1] if n_types > 1 else axes[1]
            errors = np.array(actuals) - np.array(predictions)
            ax.hist(errors, bins=50, color='#336699', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{series_type.replace("_", " ").title()}: Error Distribution\n'
                        f'MAE={res["mae"]:.2f}, MSE={res["mse"]:.2f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "timeseries.pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
        print(f"✓ Generated: paper/tables/timeseries_summary.tex")

    def run(self):
        """Run time-series forecasting on multiple series types."""
        print("[timeseries] Starting experiment...")
        
        for series_type in self.series_types:
            print(f"[timeseries] Processing: {series_type}")
            
            # Generate series
            series = SyntheticTimeSeries(
                length=self.length,
                series_type=series_type,
                seed=42,
            )
            
            train_bytes, test_bytes = series.train_test_split(test_ratio=0.2)
            print(f"[timeseries] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
            
            # Train manifold
            tokenizer_config = TokenizerConfig(
                hash_vocab_size=4096,
                hash_prime=31,
            )
            
            manifold = Manifold(
                SimulationConfig(
                    dashboard=False,
                    generator=lambda tb=train_bytes: (bytes([b]) for b in tb),
                    geometric=GeometricSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                    ),
                    spectral=SpectralSimulationConfig(
                        grid_size=(32, 32, 32),
                        dt=0.01,
                    ),
                    tokenizer=tokenizer_config,
                    position_init="random",
                    position_init_seed=42,
                ),
                observers={
                    "spectral": InferenceObserver([CarrierObserver(None)])
                }
            )
            
            state = manifold.run()
            
            # Build dual-domain inference
            geo_state = {
                "positions": state.get("positions"),
                "velocities": state.get("velocities"),
                "energies": state.get("energies"),
                "heats": state.get("heats"),
                "excitations": state.get("excitations"),
                "token_ids": state.get("token_ids"),
                "masses": state.get("masses"),
            }
            carriers = manifold.carriers or {}
            
            predictor = TimeSeriesPredictor(
                vocab_size=4096, prime=31, period_samples=period_samples
            )
            predictor.learn_from_manifold(geo_state, carriers)
            
            n_particles = geo_state["token_ids"].numel() if geo_state["token_ids"] is not None else 0
            
            # Test predictions
            actuals = []
            predictions = []
            
            # Predict on test set
            full_data = train_bytes + test_bytes
            train_len = len(train_bytes)
            
            for pos in range(train_len, len(full_data)):
                actual = full_data[pos]
                
                # Context: last context_length particles from training
                context_start = max(0, n_particles - self.context_length)
                context_indices = torch.arange(
                    context_start, n_particles,
                    device=predictor.inference.device if predictor.inference else torch.device("cpu")
                )
                
                predicted = predictor.predict(context_indices, pos)
                
                actuals.append(actual)
                predictions.append(predicted)
            
            # Calculate metrics
            actuals_np = np.array(actuals, dtype=np.float32)
            predictions_np = np.array(predictions, dtype=np.float32)
            
            mae = np.mean(np.abs(actuals_np - predictions_np))
            mse = np.mean((actuals_np - predictions_np) ** 2)
            
            # Accuracy: exact match (harsh but interpretable)
            exact_match = np.mean(actuals_np == predictions_np)
            
            # Direction accuracy (did we predict increase/decrease correctly?)
            if len(actuals_np) > 1:
                actual_diff = np.diff(actuals_np)
                pred_diff = np.diff(predictions_np)
                direction_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
            else:
                direction_accuracy = 0.0
            
            self.results[series_type] = {
                "mae": float(mae),
                "mse": float(mse),
                "accuracy": float(exact_match),
                "direction_accuracy": float(direction_accuracy),
                "actuals": actuals,
                "predictions": predictions,
            }
            
            print(f"[timeseries] {series_type}: MAE={mae:.2f}, MSE={mse:.2f}, "
                  f"Accuracy={exact_match:.3f}, Direction={direction_accuracy:.3f}")
        
        self.observe(state)
        print("[timeseries] Experiment complete.")
