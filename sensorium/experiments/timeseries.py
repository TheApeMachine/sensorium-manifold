"""Time Series Prediction Experiment

Uses the ETT (Electricity Transformer Temperature) dataset from HuggingFace.
This is a standard benchmark for time series forecasting.

Goal: Predict future values from context window.
Metrics: MSE, MAE
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.nn import functional as F

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.spectral.unified import UnifiedManifold, Modality

from .base import BaseExperiment, Scale, ScaleConfig


class TimeSeriesExperiment(BaseExperiment):
    """Time series prediction using thermodynamic dynamics.
    
    The approach:
    1. Treat each time step as a particle in 1D "value space"
    2. Context window creates attractors
    3. Predict by letting a query particle diffuse toward attractors
    4. Read out the final position as the predicted value
    """
    
    name = "timeseries"
    goal = "Predict future values from historical context"
    
    # Dataset config
    dataset_name = "ettm1"  # ETT-small dataset
    context_length = 96     # ~4 days at 15-min intervals
    prediction_length = 24  # ~1 day ahead
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
        feature: str = "OT",  # Oil Temperature (main target)
    ):
        super().__init__(scale, device, seed)
        self.feature = feature
        
        # Scale-specific configs
        if scale == Scale.TOY:
            self.context_length = 24
            self.prediction_length = 6
        elif scale == Scale.MEDIUM:
            self.context_length = 48
            self.prediction_length = 12
        else:
            self.context_length = 96
            self.prediction_length = 24
    
    def setup(self) -> None:
        """Load ETT dataset and initialize model."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets: pip install datasets"
            )
        
        print(f"    Loading electricity dataset (streaming)...")
        
        # Use chronos_datasets which has proper parquet format
        # The 'monash_electricity_hourly' subset has hourly electricity consumption
        try:
            dataset = load_dataset(
                "autogluon/chronos_datasets",
                "monash_electricity_hourly",
                streaming=True,
            )
            self._use_real_data = True
            
            # Prepare data iterators
            self.train_stream = dataset["train"]
            
            # Buffer for streaming
            self._train_buffer: List[Dict] = []
            self._eval_buffer: List[Dict] = []
            self._train_iter = iter(self.train_stream)
            
            # Prefetch some data - each row is a full time series
            self._prefetch_buffer(self._train_buffer, self._train_iter, 100)
            
            # Extract values from the first time series
            if self._train_buffer and "target" in self._train_buffer[0]:
                # chronos format has 'target' as list of values
                all_values = []
                for row in self._train_buffer:
                    target = row.get("target", [])
                    if isinstance(target, list):
                        all_values.extend(target)
                    elif hasattr(target, "tolist"):
                        all_values.extend(target.tolist())
                
                if all_values:
                    self._mean = sum(all_values) / len(all_values)
                    self._std = (sum((v - self._mean)**2 for v in all_values) / len(all_values)) ** 0.5
                else:
                    self._mean, self._std = 0.0, 1.0
            else:
                self._mean, self._std = 0.0, 1.0
                
        except Exception as e:
            print(f"    Could not load real data: {e}")
            print(f"    Falling back to synthetic data")
            self._use_real_data = False
            self._train_buffer = []
            self._eval_buffer = []
            self._mean, self._std = 0.0, 1.0
        
        self._std = max(self._std, 1e-8)
        
        # Flatten time series data into single array
        self._train_values: List[float] = []
        self._eval_values: List[float] = []
        
        if self._use_real_data and self._train_buffer:
            for row in self._train_buffer:
                target = row.get("target", [])
                if isinstance(target, list):
                    self._train_values.extend(target)
                elif hasattr(target, "tolist"):
                    self._train_values.extend(target.tolist())
            
            # Use last 20% as eval
            split_idx = int(len(self._train_values) * 0.8)
            self._eval_values = self._train_values[split_idx:]
            self._train_values = self._train_values[:split_idx]
        else:
            # Generate synthetic data
            self._train_values = self._generate_synthetic_series(5000)
            self._eval_values = self._generate_synthetic_series(1000)
        
        print(f"    Using real data: {self._use_real_data}")
        print(f"    Train values: {len(self._train_values)}")
        print(f"    Eval values: {len(self._eval_values)}")
        print(f"    Mean: {self._mean:.4f}, Std: {self._std:.4f}")
        print(f"    Context: {self.context_length}, Prediction: {self.prediction_length}")
        
        # Initialize manifold
        self.manifold = UnifiedManifold(
            self.physics_config,
            self.device,
            embed_dim=self.scale_config.embed_dim,
        )
    
    def _prefetch_buffer(
        self, 
        buffer: List[Dict], 
        iterator: Iterator,
        count: int,
    ) -> None:
        """Prefetch data into buffer."""
        for _ in range(count):
            try:
                buffer.append(next(iterator))
            except StopIteration:
                break
    
    def _generate_synthetic_series(self, length: int) -> List[float]:
        """Generate synthetic time series for testing."""
        import math
        values = []
        for t in range(length):
            # Mix of trends, seasonality, and noise
            trend = 0.001 * t
            daily = math.sin(2 * math.pi * t / 24) * 0.5
            weekly = math.sin(2 * math.pi * t / (24 * 7)) * 0.3
            noise = (torch.randn(1).item()) * 0.1
            values.append(trend + daily + weekly + noise)
        return values
    
    def _normalize(self, value: float) -> float:
        return (value - self._mean) / self._std
    
    def _denormalize(self, value: float) -> float:
        return value * self._std + self._mean
    
    def _get_window(self, values: List[float], start: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract context and target windows from flat value list."""
        total_len = self.context_length + self.prediction_length
        
        if start + total_len > len(values):
            return None
        
        context = torch.tensor([
            self._normalize(values[start + i])
            for i in range(self.context_length)
        ], dtype=torch.float32, device=self.device)
        
        target = torch.tensor([
            self._normalize(values[start + self.context_length + i])
            for i in range(self.prediction_length)
        ], dtype=torch.float32, device=self.device)
        
        return context, target
    
    def train_iterator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over training windows."""
        max_samples = self.scale_config.max_train_samples or len(self._train_values)
        total_len = self.context_length + self.prediction_length
        
        # Sliding windows over the flat value array
        for start in range(0, min(max_samples, len(self._train_values) - total_len)):
            window = self._get_window(self._train_values, start)
            if window is not None:
                yield window
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """One step of thermodynamic time series learning."""
        context, target = batch
        
        # Clear manifold
        self.manifold.clear()
        
        # Encode context as particles
        # Position = (time_index, value)
        for t, value in enumerate(context):
            position = torch.tensor([float(t), value.item()], device=self.device)
            self.manifold.add_particle(
                position=position,
                energy=1.0 / len(context),
                modality=Modality.UNKNOWN,
            )
        
        # Add attractors at context positions (for diffusion)
        for t, value in enumerate(context):
            position = torch.tensor([float(t), value.item()], device=self.device)
            self.manifold.add_attractor(
                position=position,
                energy=1.0 / len(context),
            )
        
        # Run thermodynamic dynamics
        for _ in range(5):
            self.manifold.step()
        
        # Predict: add query particles at future time indices
        predictions = []
        for t in range(self.prediction_length):
            query_time = self.context_length + t
            
            # Initialize query at extrapolated position
            if len(context) >= 2:
                # Simple linear extrapolation for initial position
                slope = (context[-1] - context[-2]).item()
                init_value = context[-1].item() + slope * (t + 1)
            else:
                init_value = context[-1].item()
            
            query_pos = torch.tensor([float(query_time), init_value], device=self.device)
            self.manifold.add_particle(
                position=query_pos,
                energy=1.0,
                modality=Modality.UNKNOWN,
            )
        
        # Let queries diffuse
        for _ in range(10):
            self.manifold.step()
        
        # Read out predictions (from last prediction_length particles)
        for i in range(self.prediction_length):
            idx = len(context) + i
            if idx < len(self.manifold._particles):
                pred_value = self.manifold._particles[idx].position[1].item()
                predictions.append(pred_value)
            else:
                predictions.append(0.0)
        
        predictions = torch.tensor(predictions, device=self.device)
        
        # Compute loss (for monitoring, not for gradients!)
        mse = F.mse_loss(predictions, target).item()
        mae = F.l1_loss(predictions, target).item()
        
        return {"mse": mse, "mae": mae}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on eval set."""
        mse_total = 0.0
        mae_total = 0.0
        count = 0
        
        total_len = self.context_length + self.prediction_length
        max_samples = self.scale_config.max_eval_samples or len(self._eval_values)
        
        # Step by total_len to get non-overlapping windows
        for start in range(0, min(max_samples, len(self._eval_values) - total_len), total_len):
            window = self._get_window(self._eval_values, start)
            if window is None:
                break
            
            context, target = window
            
            # Same as train_step but without recording history
            self.manifold.clear()
            
            for t, value in enumerate(context):
                position = torch.tensor([float(t), value.item()], device=self.device)
                self.manifold.add_particle(position=position, energy=1.0/len(context))
                self.manifold.add_attractor(position=position, energy=1.0/len(context))
            
            for _ in range(5):
                self.manifold.step()
            
            predictions = []
            for t in range(self.prediction_length):
                query_time = self.context_length + t
                if len(context) >= 2:
                    slope = (context[-1] - context[-2]).item()
                    init_value = context[-1].item() + slope * (t + 1)
                else:
                    init_value = context[-1].item()
                
                query_pos = torch.tensor([float(query_time), init_value], device=self.device)
                self.manifold.add_particle(position=query_pos, energy=1.0)
            
            for _ in range(10):
                self.manifold.step()
            
            for i in range(self.prediction_length):
                idx = len(context) + i
                if idx < len(self.manifold._particles):
                    predictions.append(self.manifold._particles[idx].position[1].item())
                else:
                    predictions.append(0.0)
            
            predictions = torch.tensor(predictions, device=self.device)
            
            mse_total += F.mse_loss(predictions, target).item()
            mae_total += F.l1_loss(predictions, target).item()
            count += 1
        
        if count == 0:
            return {"mse": float("nan"), "mae": float("nan")}
        
        return {
            "mse": mse_total / count,
            "mae": mae_total / count,
            "rmse": (mse_total / count) ** 0.5,
            "eval_samples": count,
        }


def run_timeseries_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = TimeSeriesExperiment(scale=scale, device=device)
    result = exp.run()
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
    }
