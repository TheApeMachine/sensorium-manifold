"""Dataset module for the Sensorium Manifold.

Provides a unified taxonomy of dataset types:

1. FilesystemDataset - Load raw bytes from files/directories
2. SyntheticDataset - Generate controlled byte patterns  
3. SpectralDataset - FFT-based spectral analysis for audio

All datasets implement the DatasetProtocol, yielding (byte_value, sequence_index)
tuples for the universal tokenizer.

Examples:
    # Load files from a directory
    from sensorium.dataset import FilesystemDataset, FilesystemConfig
    
    dataset = FilesystemDataset(FilesystemConfig(
        path=Path("./data/texts"),
    ))
    
    # Generate synthetic collision data
    from sensorium.dataset import SyntheticDataset, SyntheticConfig, SyntheticPattern
    
    dataset = SyntheticDataset(SyntheticConfig(
        pattern=SyntheticPattern.COLLISION,
        num_units=20,
        unit_length=32,
        collision_rate=0.5,
    ))
    
    # Spectral analysis for audio
    from sensorium.dataset import SpectralDataset, SpectralConfig
    
    dataset = SpectralDataset(SpectralConfig(
        path=Path("./audio/mixed.wav"),
        fft_size=1024,
    ))
"""

# Base types and protocol
from sensorium.dataset.base import (
    # Protocol
    DatasetProtocol,
    BaseDataset,
    
    # Configurations
    SyntheticConfig,
    SyntheticPattern,
    FilesystemConfig,
    SpectralConfig,
    
    # Utilities
    BytesDataset,
    bytes_to_tuples,
    stream_file,
    stream_files,
)

# Dataset implementations
from sensorium.dataset.filesystem import FilesystemDataset
from sensorium.dataset.synthetic import SyntheticDataset
from sensorium.dataset.spectral import SpectralDataset

# MNIST utilities
from sensorium.dataset.mnist_idx import iter_images as mnist_iter_images

# HuggingFace streaming
try:
    from sensorium.dataset.huggingface import (
        HuggingFaceConfig,
        HuggingFaceDataset,
        fineweb_edu,
        the_pile,
        wikipedia,
    )
except Exception:  # pragma: no cover - optional dependency surface
    pass

# Time series
try:
    from sensorium.dataset.timeseries import (
        TimeSeriesConfig,
        TimeSeriesDataset,
        quantize_to_bytes,
    )
except Exception:  # pragma: no cover
    pass

# Text
try:
    from sensorium.dataset.text import (
        TextDatasetConfig,
        TextDataset,
    )
except Exception:  # pragma: no cover
    pass

# Diffusion
try:
    from sensorium.dataset.diffusion import (
        DiffusionDatasetConfig,
        DiffusionDataset,
    )
except Exception:  # pragma: no cover
    pass

# Audio
try:
    from sensorium.dataset.audio import (
        AudioDatasetConfig,
        AudioDataset,
        quantize_audio_to_bytes,
        dequantize_bytes_to_audio,
    )
except Exception:  # pragma: no cover
    pass

# Rule shift
try:
    from sensorium.dataset.rule_shift import (
        RuleShiftConfig,
        RuleShiftDataset,
    )
except Exception:  # pragma: no cover
    pass

# Cross-modal
try:
    from sensorium.dataset.cross_modal import (
        CrossModalConfig,
        CrossModalDataset,
        create_stripe_image,
        create_checkerboard_image,
    )
except Exception:  # pragma: no cover
    pass

# MNIST
try:
    from sensorium.dataset.mnist import (
        MNISTConfig,
        MNISTDataset,
        MNIST_IMAGE_SIZE,
    )
except Exception:  # pragma: no cover
    pass

# Cocktail party
try:
    from sensorium.dataset.cocktail_party import (
        CocktailPartyConfig,
        CocktailPartyDataset,
        save_wav,
    )
except Exception:  # pragma: no cover
    pass

# Scaling
try:
    from sensorium.dataset.scaling import (
        ScalingDatasetConfig,
        ScalingDataset,
        ScalingTestType,
        GeneralizationType,
    )
except Exception:  # pragma: no cover
    pass


__all__ = [
    # Protocol and base
    "DatasetProtocol",
    "BaseDataset",
    
    # Configurations
    "SyntheticConfig",
    "SyntheticPattern",
    "FilesystemConfig",
    "SpectralConfig",
    
    # Datasets
    "FilesystemDataset",
    "SyntheticDataset",
    "SpectralDataset",
    
    # Utilities
    "BytesDataset",
    "bytes_to_tuples",
    "stream_file",
    "stream_files",
    "mnist_iter_images",
    
    # HuggingFace
    "HuggingFaceConfig",
    "HuggingFaceDataset",
    "HuggingFaceDatasetConfig",
    "fineweb_edu",
    "the_pile",
    "wikipedia",
    
    # Time series
    "TimeSeriesConfig",
    "TimeSeriesDataset",
    "quantize_to_bytes",
    
    # Text
    "TextDatasetConfig",
    "TextDataset",
    
    # Diffusion
    "DiffusionDatasetConfig",
    "DiffusionDataset",
    
    # Audio
    "AudioDatasetConfig",
    "AudioDataset",
    "quantize_audio_to_bytes",
    "dequantize_bytes_to_audio",
    
    # Rule shift
    "RuleShiftConfig",
    "RuleShiftDataset",
    
    # Cross-modal
    "CrossModalConfig",
    "CrossModalDataset",
    "create_stripe_image",
    "create_checkerboard_image",
    
    # MNIST
    "MNISTConfig",
    "MNISTDataset",
    "MNIST_IMAGE_SIZE",
    
    # Cocktail party
    "CocktailPartyConfig",
    "CocktailPartyDataset",
    "save_wav",
    
    # Scaling
    "ScalingDatasetConfig",
    "ScalingDataset",
    "ScalingTestType",
    "GeneralizationType",
]

# Keep star-import surface consistent when optional imports are unavailable.
__all__ = [name for name in __all__ if name in globals()]
