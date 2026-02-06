"""Kernel time-series forecasting (byte-quantized).

This experiment uses the clean composable pattern:
- Datasets: TimeSeriesDataset (quantizes signals to bytes)
- Observers: InferenceObserver with PositionPeriodicityPredictor
- Projectors: Config-driven tables and figures

Produces:
- `paper/tables/timeseries_summary.tex`
- `paper/figures/timeseries.png`
"""

from __future__ import annotations

from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig, CoherenceSimulationConfig
from optimizer.tokenizer import TokenizerConfig

# Datasets
from sensorium.dataset import TimeSeriesDataset, TimeSeriesConfig

# Observers
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.metrics import (
    PositionPeriodicityPredictor,
    ParticleCount,
    ModeCount,
)

# Projectors
from sensorium.projectors import (
    PipelineProjector,
    ConsoleProjector,
    LaTeXTableProjector,
    FigureProjector,
    TableConfig,
    FigureConfig,
)


class KernelTimeSeries(Experiment):
    """Time-series forecasting experiment using byte-quantized values.
    
    Clean pattern:
    - datasets: TimeSeriesDataset for each signal type
    - manifold: Runs simulation
    - inference: InferenceObserver accumulates results across series types
    - projector: Config-driven outputs query InferenceObserver
    """
    
    def __init__(self, experiment_name: str, profile: bool = False, dashboard: bool = False):
        super().__init__(experiment_name, profile, dashboard=dashboard)
        
        # Configuration
        self.vocab_size = 4096
        self.prime = 31
        self.series_types = ["periodic", "trend_seasonal", "regime_switch", "sawtooth"]
        
        # 1. DATASETS - one for each series type
        self.datasets = {
            series_type: TimeSeriesDataset(TimeSeriesConfig(
                length=2000,
                series_type=series_type,
                segment_size=50,
                seed=42,
            ))
            for series_type in self.series_types
        }
        
        # 2. MANIFOLD - configured per series type in run()
        self.manifold = None
        
        # 3. OBSERVERS - InferenceObserver accumulates results
        self.inference = InferenceObserver([
            PositionPeriodicityPredictor(segment_size=50),
            ParticleCount(),
            ModeCount(),
        ])
        
        # 4. PROJECTORS - config-driven, query InferenceObserver
        self.projector = PipelineProjector(
            ConsoleProjector(
                fields=["series_type", "mae", "rmse", "direction_accuracy", "within_5"],
                format="table",
            ),
            LaTeXTableProjector(
                TableConfig(
                    name="timeseries_summary",
                    columns=["series_type", "mae", "rmse", "direction_accuracy", "within_5", "within_10"],
                    headers={
                        "series_type": "Series Type",
                        "mae": "MAE",
                        "rmse": "RMSE",
                        "direction_accuracy": "Direction",
                        "within_5": "Within-5",
                        "within_10": "Within-10",
                    },
                    caption="Time-series forecasting results across signal types",
                    label="tab:timeseries",
                    precision=2,
                ),
                output_dir=self.artifact_path("tables"),
            ),
            FigureProjector(
                FigureConfig(
                    name="timeseries_direction",
                    chart_type="bar",
                    x="series_type",
                    y=["direction_accuracy"],
                    title="Direction Accuracy by Series Type",
                    xlabel="Series Type",
                    ylabel="Direction Accuracy",
                    grid=True,
                ),
                output_dir=self.artifact_path("figures"),
            ),
        )

    def run(self):
        """Run the time series forecasting experiment."""
        print("[timeseries] Starting experiment...")
        
        for series_type in self.series_types:
            print(f"\n{'='*60}")
            print(f"Processing: {series_type}")
            print('='*60)
            
            dataset = self.datasets[series_type]
            print(f"  Train: {len(dataset.train_bytes)}, Test: {len(dataset.test_bytes)}")
            
            # 2. MANIFOLD - fresh instance per series type
            self.manifold = Manifold(
                SimulationConfig(
                    dashboard=self.dashboard,
                    video_path=self.video_path,
                    tokenizer=TokenizerConfig(
                        hash_vocab_size=self.vocab_size,
                        hash_prime=self.prime,
                    ),
                    coherence=CoherenceSimulationConfig(
                        max_carriers=64,
                        stable_amp_threshold=0.15,
                        crystallize_amp_threshold=0.20,
                        grid_size=(64, 64, 64),
                        dt=0.01,
                    ),
                )
            )
            
            # Run simulation
            self.manifold.add_dataset(dataset.generate)
            state = self.manifold.run()
            
            # Observe - pass dataset info and metadata
            # InferenceObserver accumulates results
            self.inference.observe(
                {
                    **state,
                    "train_bytes": dataset.train_bytes,
                    "test_bytes": dataset.test_bytes,
                    "split_idx": dataset.split_idx,
                },
                manifold=self.manifold,
                series_type=series_type,
            )
        
        # Project all accumulated results
        self.project()
        
        print("\n[timeseries] Experiment complete.")

    def observe(self, state: dict) -> dict:
        """Observe the manifold state using composed observers."""
        return self.inference.observe(state, manifold=self.manifold)

    def project(self) -> dict:
        """Project accumulated observations to artifacts."""
        return self.projector.project(self.inference)
