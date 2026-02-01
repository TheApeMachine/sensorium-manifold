"""Base experiment class for all experiments.

This provides the common infrastructure for:
- Scale management (toy, medium, full)
- HuggingFace dataset streaming
- Training loops (thermodynamic style)
- Metric collection
- Result formatting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.config import PhysicsConfig


class Scale(Enum):
    """Experiment scale levels."""
    TOY = "toy"
    MEDIUM = "medium"
    FULL = "full"


@dataclass
class ScaleConfig:
    """Configuration for a specific scale."""
    name: str
    # Data limits
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    # Model size (if applicable)
    embed_dim: int
    # Training
    train_steps: int
    eval_every: int
    # Physics
    dt: float = 0.02
    
    
# Default scale configurations
SCALE_CONFIGS = {
    Scale.TOY: ScaleConfig(
        name="toy",
        max_train_samples=1000,
        max_eval_samples=100,
        embed_dim=64,
        train_steps=500,
        eval_every=100,
    ),
    Scale.MEDIUM: ScaleConfig(
        name="medium",
        max_train_samples=10000,
        max_eval_samples=1000,
        embed_dim=256,
        train_steps=5000,
        eval_every=500,
    ),
    Scale.FULL: ScaleConfig(
        name="full",
        max_train_samples=None,  # Use full dataset
        max_eval_samples=5000,
        embed_dim=512,
        train_steps=50000,
        eval_every=1000,
    ),
}


@dataclass
class TrainingState:
    """Tracks training progress."""
    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    
    def record(self, name: str, value: float) -> None:
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append(value)
    
    def get_latest(self, name: str, default: float = 0.0) -> float:
        if name in self.metrics_history and self.metrics_history[name]:
            return self.metrics_history[name][-1]
        return default
    
    def get_mean(self, name: str, window: int = 100) -> float:
        if name not in self.metrics_history:
            return 0.0
        values = self.metrics_history[name][-window:]
        return sum(values) / len(values) if values else 0.0


@dataclass 
class ExperimentResult:
    """Container for experiment outputs."""
    name: str
    scale: str
    goal: str
    success: bool
    metrics: Dict[str, Any]
    failure_reason: Optional[str] = None
    tables: Dict[str, str] = field(default_factory=dict)
    figures: Dict[str, Path] = field(default_factory=dict)
    
    def summary(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED: {self.failure_reason}"
        lines = [
            f"Experiment: {self.name} ({self.scale})",
            f"Goal: {self.goal}",
            f"Status: {status}",
            "Metrics:",
        ]
        for k, v in self.metrics.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class BaseExperiment(ABC):
    """Base class for all experiments.
    
    Subclasses should implement:
    - setup(): Initialize model, load data
    - train_step(): One step of thermodynamic training
    - evaluate(): Compute metrics on eval set
    - cleanup(): Optional cleanup
    """
    
    name: str = "base"
    goal: str = "Base experiment"
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.scale = scale
        self.scale_config = SCALE_CONFIGS[scale]
        self.device = device or self._get_device()
        self.seed = seed
        
        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Training state
        self.state = TrainingState()
        
        # Physics config
        self.physics_config = PhysicsConfig(
            dt=self.scale_config.dt,
            eps=1e-8,
        )
        
        # Will be set by subclasses
        self.model = None
        self.train_data = None
        self.eval_data = None
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Prefer MPS on Apple Silicon when torch_scatter isn't installed.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                import torch_scatter  # type: ignore  # noqa: F401

                has_torch_scatter = True
            except Exception:
                has_torch_scatter = False
            if not has_torch_scatter:
                return torch.device("mps")
        return torch.device("cpu")
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize model and load data."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute one training step.
        
        In thermodynamic training, this typically means:
        1. Ingest context
        2. Run grammar step / physics
        3. Observe next token / target
        4. Optionally run idle pondering
        
        Returns dict of metrics for this step.
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the eval set.
        
        Returns dict of evaluation metrics.
        """
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup after experiment."""
        pass
    
    def train_iterator(self) -> Iterator[Any]:
        """Iterate over training data.
        
        Override if you need custom batching.
        """
        if self.train_data is None:
            return iter([])
        return iter(self.train_data)
    
    def _format_metrics(self, metrics: Dict[str, float], max_items: int = 4) -> str:
        """Format metrics for display."""
        items = []
        for k, v in list(metrics.items())[:max_items]:
            if isinstance(v, float):
                if abs(v) < 0.01 or abs(v) > 1000:
                    items.append(f"{k}={v:.2e}")
                else:
                    items.append(f"{k}={v:.4f}")
            else:
                items.append(f"{k}={v}")
        return ", ".join(items)
    
    def run(self) -> ExperimentResult:
        """Run the full experiment."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.name} ({self.scale.value})")
        print(f"Goal: {self.goal}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        try:
            # Setup
            print("\n[1] Setup...")
            self.setup()
            print(f"    Model initialized")
            print(f"    Train samples: {self.scale_config.max_train_samples or 'all'}")
            print(f"    Eval samples: {self.scale_config.max_eval_samples or 'all'}")
            
            # Training loop
            print(f"\n[2] Training ({self.scale_config.train_steps} steps)...")
            
            data_iter = self.train_iterator()
            
            # Create progress bar
            if tqdm is not None:
                pbar = tqdm(
                    range(self.scale_config.train_steps),
                    desc="Training",
                    unit="step",
                    ncols=100,
                    leave=True,
                )
            else:
                pbar = range(self.scale_config.train_steps)
            
            last_metrics: Dict[str, float] = {}
            
            for step in pbar:
                self.state.step = step
                
                # Get next batch (cycle if needed)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.state.epoch += 1
                    data_iter = self.train_iterator()
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        print("    Warning: No training data available")
                        break
                
                # Train step
                step_metrics = self.train_step(batch)
                for k, v in step_metrics.items():
                    self.state.record(k, v)
                
                self.state.samples_seen += 1
                
                # Update progress bar with metrics
                if tqdm is not None:
                    # Show rolling averages
                    display_metrics = {
                        k: self.state.get_mean(k, window=50)
                        for k in list(step_metrics.keys())[:3]
                    }
                    display_metrics["epoch"] = self.state.epoch
                    pbar.set_postfix(display_metrics)
                
                # Periodic evaluation
                if (step + 1) % self.scale_config.eval_every == 0:
                    eval_metrics = self.evaluate()
                    last_metrics = eval_metrics
                    
                    if tqdm is None:
                        # Fallback text output
                        train_summary = self._format_metrics(
                            {k: self.state.get_mean(k) for k in step_metrics.keys()}
                        )
                        eval_summary = self._format_metrics(eval_metrics)
                        print(f"    Step {step+1}: train=[{train_summary}]")
                        print(f"              eval=[{eval_summary}]")
                    else:
                        # Brief eval summary in tqdm
                        tqdm.write(f"  [Eval @ {step+1}] {self._format_metrics(eval_metrics)}")
            
            if tqdm is not None and hasattr(pbar, 'close'):
                pbar.close()
            
            # Final evaluation
            print(f"\n[3] Final evaluation...")
            final_metrics = self.evaluate()
            
            print(f"\n    {'─'*40}")
            for k, v in final_metrics.items():
                if isinstance(v, float):
                    if abs(v) < 0.01 or abs(v) > 1000:
                        print(f"    {k:.<30} {v:.4e}")
                    else:
                        print(f"    {k:.<30} {v:.4f}")
                else:
                    print(f"    {k:.<30} {v}")
            print(f"    {'─'*40}")
            
            # Cleanup
            self.cleanup()
            
            # Success
            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=True,
                metrics={
                    "final": final_metrics,
                    "training_history": self.state.metrics_history,
                    "steps": self.state.step,
                    "epochs": self.state.epoch,
                    "samples_seen": self.state.samples_seen,
                },
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"\n[FAILED] {e}")
            print(tb)
            
            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=False,
                metrics={"training_history": self.state.metrics_history},
                failure_reason=str(e),
            )
