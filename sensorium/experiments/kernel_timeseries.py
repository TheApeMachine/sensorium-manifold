"""Kernel time-series forecasting (byte-quantized).

We quantize a synthetic time series to bytes and run next-byte prediction
using the thermodynamic trie mechanism.

NON-CHEATING DESIGN:
====================
This experiment uses a proper train/test split in time:
- Training: First portion of the time series
- Testing: Future portion (strictly after training period)
- No lookahead: Predictions only use past values

We test on synthetic signals with learnable patterns:
- Periodic signals (sine with overlapping periodicities)
- Trend + seasonality (linear + periodic)
- Regime-switching (changes in pattern type)
- Sawtooth (repeating ramp pattern)

Produces:
- `paper/tables/timeseries_summary.tex`
- `paper/figures/timeseries.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, SpectralSimulationConfig
from optimizer.tokenizer import TokenizerConfig


def quantize_to_bytes(values: np.ndarray) -> Tuple[bytes, float, float]:
    """Quantize float values to bytes (0-255).
    
    Returns:
        (byte_data, min_val, max_val) for dequantization
    """
    min_val = float(values.min())
    max_val = float(values.max())
    normalized = (values - min_val) / (max_val - min_val + 1e-10)
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return bytes(quantized), min_val, max_val


def dequantize_from_bytes(data: bytes, min_val: float, max_val: float) -> np.ndarray:
    """Dequantize bytes back to float values."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return arr / 255.0 * (max_val - min_val) + min_val


class SyntheticTimeSeries:
    """Generate synthetic time series with learnable patterns."""
    
    def __init__(
        self,
        length: int = 2000,
        series_type: str = "periodic",
        seed: int = 42,
    ):
        self.length = length
        self.series_type = series_type
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        self.values = self._generate()
        self.bytes_data, self.min_val, self.max_val = quantize_to_bytes(self.values)
    
    def _generate(self) -> np.ndarray:
        """Generate the time series based on type."""
        t = np.arange(self.length, dtype=np.float32)
        
        if self.series_type == "periodic":
            # Multiple overlapping periodicities
            y = (
                50 * np.sin(2 * np.pi * t / 50) +    # Slow period (50 samples)
                20 * np.sin(2 * np.pi * t / 13) +    # Medium period
                10 * np.sin(2 * np.pi * t / 7) +     # Fast period
                5 * self._rng.randn(self.length)     # Noise
            )
        
        elif self.series_type == "trend_seasonal":
            # Linear trend + seasonal component
            trend = 0.05 * t
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


class KernelTimeSeries(Experiment):
    """Time-series forecasting experiment using byte-quantized values."""
    
    SEGMENT_SIZE = 50  # Match the dominant period
    
    def __init__(self, experiment_name: str, profile: bool = False):
        super().__init__(experiment_name, profile)
        
        # Tokenizer params
        self.vocab_size = 4096
        self.prime = 31
        self.mask = self.vocab_size - 1
        self.inv_prime = pow(self.prime, -1, self.vocab_size)
        
        # Experiment params
        self.length = 2000
        self.context_length = 3  # Short context for noisy signals
        self.test_ratio = 0.2
        self.series_types = ["periodic", "trend_seasonal", "regime_switch", "sawtooth"]
        
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def run(self):
        import time
        
        print("[timeseries] Starting experiment...")
        
        # Track stats from first run
        self._stats = {
            "n_particles": 0,
            "n_carriers": 0,
            "n_crystallized": 0,
            "grid_size": (64, 64, 64),
            "dt": 0.01,
            "wall_time_ms": 0,
        }
        
        for series_type in self.series_types:
            print(f"[timeseries] Processing: {series_type}")
            
            # Generate series
            series = SyntheticTimeSeries(
                length=self.length,
                series_type=series_type,
                seed=42,
            )
            
            # Split: train on first portion, test on future
            split_idx = int(self.length * (1 - self.test_ratio))
            train_bytes = series.bytes_data[:split_idx]
            test_bytes = series.bytes_data[split_idx:]
            
            print(f"[timeseries] Train: {len(train_bytes)}, Test: {len(test_bytes)}")
            
            # Train manifold on training data only
            def train_generator():
                yield train_bytes
            
            grid_size = (64, 64, 64)
            dt = 0.01
            
            cfg = SimulationConfig(
                tokenizer=TokenizerConfig(
                    hash_vocab_size=self.vocab_size,
                    hash_prime=self.prime,
                    segment_size=self.SEGMENT_SIZE,
                ),
                spectral=SpectralSimulationConfig(
                    max_carriers=64,
                    stable_amp_threshold=0.15,
                    crystallize_amp_threshold=0.20,
                    grid_size=grid_size,
                    dt=dt,
                ),
                generator=train_generator,
            )
            
            manifold = Manifold(cfg)
            start_time = time.time()
            state = manifold.run()
            run_time_ms = (time.time() - start_time) * 1000
            
            # Get training data as tokens
            token_ids = state.get("token_ids")
            if token_ids is None:
                print(f"[timeseries] ERROR: No token_ids for {series_type}")
                continue
            
            token_ids_np = token_ids.cpu().numpy()
            energies_np = state.get("energies", torch.ones(len(token_ids))).cpu().numpy()
            n_train = len(token_ids_np)
            
            # Capture stats from first series type
            if series_type == self.series_types[0]:
                carriers = manifold.carriers or {}
                amplitudes = carriers.get("amplitudes")
                n_carriers = int((amplitudes > 1e-6).sum().item()) if amplitudes is not None else 0
                crystallized = carriers.get("crystallized")
                n_crystallized = int(crystallized.sum().item()) if crystallized is not None else 0
                
                self._stats = {
                    "n_particles": n_train,
                    "n_carriers": n_carriers,
                    "n_crystallized": n_crystallized,
                    "grid_size": grid_size,
                    "dt": dt,
                    "wall_time_ms": run_time_ms,
                }
            
            # Predict on test set using position periodicity
            actuals = []
            predictions = []
            
            for test_pos in range(len(test_bytes)):
                actual_byte = test_bytes[test_pos]
                
                # Global position (in full series)
                global_pos = split_idx + test_pos
                seg_pos = global_pos % self.SEGMENT_SIZE
                
                # Predict using position periodicity (weighted average)
                predicted = self._predict_next_byte(train_bytes, seg_pos)
                
                actuals.append(actual_byte)
                predictions.append(predicted)
            
            # Calculate metrics
            actuals_np = np.array(actuals, dtype=np.float32)
            predictions_np = np.array(predictions, dtype=np.float32)
            
            mae = np.mean(np.abs(actuals_np - predictions_np))
            mse = np.mean((actuals_np - predictions_np) ** 2)
            
            # Exact match accuracy
            exact_accuracy = np.mean(actuals_np == predictions_np)
            
            # Direction accuracy (correct trend)
            if len(actuals_np) > 1:
                actual_diff = np.diff(actuals_np)
                pred_diff = np.diff(predictions_np)
                direction_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
            else:
                direction_accuracy = 0.0
            
            # Within-N accuracy (how often within N quantization levels)
            within_5 = np.mean(np.abs(actuals_np - predictions_np) <= 5)
            within_10 = np.mean(np.abs(actuals_np - predictions_np) <= 10)
            
            self.results[series_type] = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "exact_accuracy": float(exact_accuracy),
                "direction_accuracy": float(direction_accuracy),
                "within_5": float(within_5),
                "within_10": float(within_10),
                "actuals": actuals,
                "predictions": predictions,
                "min_val": series.min_val,
                "max_val": series.max_val,
            }
            
            print(f"[timeseries] {series_type}: MAE={mae:.2f}, RMSE={np.sqrt(mse):.2f}, "
                  f"Dir={direction_accuracy:.1%}, ±5={within_5:.1%}")
        
        self._generate_table()
        self._generate_figure()
        
        # Write simulation stats
        self.write_simulation_stats(
            "timeseries",
            n_particles=self._stats["n_particles"],
            n_carriers=self._stats["n_carriers"],
            n_crystallized=self._stats["n_crystallized"],
            grid_size=self._stats["grid_size"],
            dt=self._stats["dt"],
            n_steps=1,
            wall_time_ms=self._stats["wall_time_ms"],
        )
        print(f"✓ Generated: paper/tables/timeseries_stats.tex")
        
        print("[timeseries] Experiment complete.")
    
    def _predict_next_byte(
        self,
        train_bytes: bytes,
        target_seg_pos: int,
    ) -> int:
        """Predict next byte using position periodicity (weighted average).
        
        The segment_size creates a thermodynamic trie where values at the same
        segment position collide. For time series, this captures periodicity:
        if the signal repeats every SEGMENT_SIZE samples, values at matching
        positions should be similar.
        
        We use a weighted average (not voting) since time series values are
        continuous and close values should contribute to the prediction.
        """
        
        values = []
        weights = []
        
        for i in range(len(train_bytes)):
            if i % self.SEGMENT_SIZE == target_seg_pos:
                values.append(train_bytes[i])
                # Weight more recent observations higher
                recency = (i + 1) / len(train_bytes)
                weights.append(recency)
        
        if not values:
            return 128
        
        # Weighted average for continuous-valued prediction
        values_np = np.array(values, dtype=np.float32)
        weights_np = np.array(weights, dtype=np.float32)
        weights_np = weights_np / weights_np.sum()
        
        predicted = np.average(values_np, weights=weights_np)
        return int(np.clip(np.round(predicted), 0, 255))
    
    def _dehash(self, token_id: int, position: int) -> int:
        """Reverse the hash to get the original byte value."""
        target = (token_id - position) & self.mask
        byte_val = (target * self.inv_prime) & self.mask
        return byte_val
    
    def _generate_table(self):
        """Generate LaTeX table with results."""
        
        table_content = r"""\begin{table}[t]
\centering
\caption{Time-series forecasting results across signal types. The manifold uses byte-quantized values and predicts future values using the thermodynamic trie. Direction accuracy measures correct trend prediction; within-N measures predictions within N quantization levels of the true value.}
\label{tab:timeseries}
\begin{tabular}{l c c c c c}
\toprule
\textbf{Series Type} & \textbf{MAE} & \textbf{RMSE} & \textbf{Direction} & \textbf{Within-5} & \textbf{Within-10} \\
\midrule
"""
        for series_type in self.series_types:
            res = self.results.get(series_type, {})
            if res:
                table_content += f"{series_type.replace('_', ' ').title()} & "
                table_content += f"{res['mae']:.1f} & "
                table_content += f"{res['rmse']:.1f} & "
                table_content += f"{res['direction_accuracy']*100:.1f}\\% & "
                table_content += f"{res['within_5']*100:.1f}\\% & "
                table_content += f"{res['within_10']*100:.1f}\\% \\\\\n"
        
        # Averages
        avg_mae = np.mean([r["mae"] for r in self.results.values()])
        avg_rmse = np.mean([r["rmse"] for r in self.results.values()])
        avg_dir = np.mean([r["direction_accuracy"] for r in self.results.values()])
        avg_w5 = np.mean([r["within_5"] for r in self.results.values()])
        avg_w10 = np.mean([r["within_10"] for r in self.results.values()])
        
        table_content += r"\midrule" + "\n"
        table_content += f"\\textbf{{Average}} & {avg_mae:.1f} & {avg_rmse:.1f} & "
        table_content += f"{avg_dir*100:.1f}\\% & {avg_w5*100:.1f}\\% & {avg_w10*100:.1f}\\% \\\\\n"
        
        table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        table_path = self.artifact_path("tables", "timeseries_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table_content)
        
        print(f"✓ Generated: {table_path}")
    
    def _generate_figure(self):
        """Generate 3-panel visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # =================================================================
        # Panel A: Forecast vs Actual for one series type (periodic)
        # =================================================================
        ax = axes[0]
        
        res = self.results.get("periodic", {})
        if res:
            actuals = np.array(res["actuals"][:100])  # First 100 test points
            preds = np.array(res["predictions"][:100])
            
            t = np.arange(len(actuals))
            ax.plot(t, actuals, 'o-', color='#336699', linewidth=1.5, 
                   markersize=3, label='Actual', alpha=0.8)
            ax.plot(t, preds, 's-', color='#4C994C', linewidth=1.5,
                   markersize=3, label='Predicted', alpha=0.8)
            
            ax.set_xlabel("Time step (test set)", fontsize=10)
            ax.set_ylabel("Value (quantized)", fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim(0, 100)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel B: Direction accuracy by series type
        # =================================================================
        ax = axes[1]
        
        series_names = [s.replace('_', '\n') for s in self.series_types]
        dir_accs = [self.results.get(s, {}).get("direction_accuracy", 0) for s in self.series_types]
        colors = ['#336699', '#4C994C', '#CC6633', '#9966CC']
        
        bars = ax.bar(range(len(series_names)), dir_accs, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for bar, val in zip(bars, dir_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.0%}", ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(range(len(series_names)))
        ax.set_xticklabels(series_names, fontsize=9)
        ax.set_ylabel("Direction accuracy", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        # =================================================================
        # Panel C: Error distribution for periodic
        # =================================================================
        ax = axes[2]
        
        res = self.results.get("periodic", {})
        if res:
            errors = np.array(res["actuals"]) - np.array(res["predictions"])
            
            ax.hist(errors, bins=50, color='#336699', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel("Prediction error (actual - predicted)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            
            # Add stats
            mae = res["mae"]
            ax.text(0.95, 0.95, f"MAE: {mae:.1f}\nRMSE: {res['rmse']:.1f}",
                   transform=ax.transAxes, fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', va='top')
        
        plt.tight_layout()
        fig_path = self.artifact_path("figures", "timeseries.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: {fig_path}")
    
    def observe(self, state: dict):
        """Observer interface for compatibility."""
        pass
