"""Projectors for generating output artifacts.

Projectors transform InferenceObserver data into output files:
- LaTeXTableProjector: Generate .tex tables
- FigureProjector: Generate matplotlib figures
- ReconstructionProjector: Output reconstructed media

Design:
- Config-driven: Just specify what fields to extract
- Query-based: Projectors query InferenceObserver for data
- Composable: Use PipelineProjector to chain multiple outputs

Example:
    # Setup projectors with config
    projector = PipelineProjector(
        LaTeXTableProjector(TableConfig(
            name="summary",
            columns=["rate", "accuracy", "loss"],
        )),
        FigureProjector(FigureConfig(
            name="metrics",
            x="rate",
            y=["accuracy", "loss"],
        )),
    )
    
    # Project from inference observer
    projector.project(inference_observer)
"""

from sensorium.projectors.base import (
    ProjectorProtocol,
    BaseProjector,
    PipelineProjector,
    ConsoleProjector,
)
from sensorium.projectors.latex import (
    TableConfig,
    LaTeXTableProjector,
)
from sensorium.projectors.figure import (
    FigureConfig,
    FigureProjector,
)
from sensorium.projectors.reconstruction import (
    ReconstructionConfig,
    ReconstructionProjector,
)
from sensorium.projectors.next_token import (
    NextTokenFigureConfig,
    NextTokenFigureProjector,
)
from sensorium.projectors.diffusion import (
    DiffusionFigureConfig,
    DiffusionTableProjector,
    DiffusionFigureProjector,
)
from sensorium.projectors.audio import (
    AudioFigureConfig,
    AudioTableProjector,
    AudioFigureProjector,
)
from sensorium.projectors.rule_shift import (
    RuleShiftFigureConfig,
    RuleShiftTableProjector,
    RuleShiftFigureProjector,
)
from sensorium.projectors.cross_modal import (
    CrossModalFigureConfig,
    CrossModalFigureProjector,
    CrossModalTableProjector,
)
from sensorium.projectors.scaling import (
    ScalingFigureConfig,
    ScalingTableProjector,
    ScalingDynamicsFigureProjector,
    ScalingComputeFigureProjector,
)
from sensorium.projectors.image import (
    ImageFigureConfig,
    ImageTableProjector,
    ImageFigureProjector,
)
from sensorium.projectors.cocktail_party import (
    CocktailPartyFigureConfig,
    CocktailPartyTableProjector,
    CocktailPartyFigureProjector,
)
from sensorium.projectors.transitions import (
    TopTransitionsConfig,
    TopTransitionsProjector,
)
from sensorium.projectors.omega_labels import (
    OmegaLabelFigureConfig,
    OmegaLabelFigureProjector,
)

__all__ = [
    # Protocols and base
    "ProjectorProtocol",
    "BaseProjector",
    "PipelineProjector",
    "ConsoleProjector",
    # LaTeX
    "TableConfig",
    "LaTeXTableProjector",
    # Figures
    "FigureConfig",
    "FigureProjector",
    # Reconstruction
    "ReconstructionConfig",
    "ReconstructionProjector",
    # Next token
    "NextTokenFigureConfig",
    "NextTokenFigureProjector",
    # Diffusion
    "DiffusionFigureConfig",
    "DiffusionTableProjector",
    "DiffusionFigureProjector",
    # Audio
    "AudioFigureConfig",
    "AudioTableProjector",
    "AudioFigureProjector",
    # Rule shift
    "RuleShiftFigureConfig",
    "RuleShiftTableProjector",
    "RuleShiftFigureProjector",
    # Cross-modal
    "CrossModalFigureConfig",
    "CrossModalFigureProjector",
    "CrossModalTableProjector",
    # Scaling
    "ScalingFigureConfig",
    "ScalingTableProjector",
    "ScalingDynamicsFigureProjector",
    "ScalingComputeFigureProjector",
    # Image
    "ImageFigureConfig",
    "ImageTableProjector",
    "ImageFigureProjector",
    # Cocktail party
    "CocktailPartyFigureConfig",
    "CocktailPartyTableProjector",
    "CocktailPartyFigureProjector",
    # Transitions artifacts
    "TopTransitionsConfig",
    "TopTransitionsProjector",
    # Labelled ω figures
    "OmegaLabelFigureConfig",
    "OmegaLabelFigureProjector",
]
