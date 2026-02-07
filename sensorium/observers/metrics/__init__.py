"""Metric observers that compute specific measurements from simulation state.

These are higher-level observers that compose primitives and transforms
to compute specific metrics needed for experiments.
"""

from .clustering import SpatialClustering, ClusteringBaseline, ClusteringExcess
from .entropy import TokenEntropy, NormalizedEntropy
from .compression import (
    CompressionRatio,
    CollisionCount,
    CollisionRate,
    CollidingTokens,
)
from .energy import EnergyAccumulation, EnergyCorrelation, MeanParticleEnergy
from .heatmap import CollisionMatrix, TokenPositionDensity
from .counts import ParticleCount, ModeCount, CrystallizedCount, UniqueTokenCount
from .periodicity import PositionPeriodicityPredictor
from .prediction import CrystallizationObserver, NextTokenMetrics
from .diffusion import TriePatternMatcher
from .audio import AudioPeriodicityPredictor
from .rule_shift import RuleShiftPredictor
from .cross_modal import CrossModalDynamicsObserver, ImageReconstructor
from .scaling import ModeTrackingObserver
from .image import ImageInpainter
from .cocktail_party import SpectralClusteringObserver
from .dehash import DehashObserver
from .map_path import MapPathMetrics, KeySpec
from .token_stats import TokenDistributionMetrics
from .wave_stats import WaveFieldMetrics
from .collision_paper import CollisionPaperArtifactsConfig, CollisionPaperArtifacts
from .omega_labels import (
    OmegaLabelProbeConfig,
    OmegaLabelProbe,
    CoherenceSpectrumSnapshot,
)

__all__ = [
    # Counts
    "ParticleCount",
    "ModeCount",
    "CrystallizedCount",
    "UniqueTokenCount",
    # Clustering
    "SpatialClustering",
    "ClusteringBaseline",
    "ClusteringExcess",
    # Entropy
    "TokenEntropy",
    "NormalizedEntropy",
    # Compression
    "CompressionRatio",
    "CollisionCount",
    "CollisionRate",
    "CollidingTokens",
    # Energy
    "EnergyAccumulation",
    "EnergyCorrelation",
    "MeanParticleEnergy",
    # Heatmap
    "CollisionMatrix",
    "TokenPositionDensity",
    # Periodicity
    "PositionPeriodicityPredictor",
    # Prediction
    "CrystallizationObserver",
    "NextTokenMetrics",
    # Diffusion
    "TriePatternMatcher",
    # Audio
    "AudioPeriodicityPredictor",
    # Rule shift
    "RuleShiftPredictor",
    # Cross-modal
    "CrossModalDynamicsObserver",
    "ImageReconstructor",
    # Scaling
    "ModeTrackingObserver",
    # Image
    "ImageInpainter",
    # Cocktail party
    "SpectralClusteringObserver",
    # Dehash
    "DehashObserver",
    # Map vs path
    "KeySpec",
    "MapPathMetrics",
    # Token stats
    "TokenDistributionMetrics",
    # Wave stats
    "WaveFieldMetrics",
    # Collision paper artifacts
    "CollisionPaperArtifactsConfig",
    "CollisionPaperArtifacts",
    # Labelled ω probes
    "OmegaLabelProbeConfig",
    "OmegaLabelProbe",
    "CoherenceSpectrumSnapshot",
]
