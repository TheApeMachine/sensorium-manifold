"""Base class and protocol for all datasets.

The Sensorium Manifold uses a "universal tokenizer" approach where data
is represented as bytes with sequence indices. This is the native format
that the thermodynamic system understands.

Dataset Taxonomy:
=================
1. FilesystemDataset - Read raw bytes from files/directories
2. SyntheticDataset - Generate synthetic byte patterns
3. SpectralDataset - Transform data through FFT before tokenization

Key concepts:
- Data is yielded as (byte_value, sequence_index) tuples
- Sequence index resets at logical boundaries (e.g., new image, new sentence)
- This enables the system to learn structure at the byte level
- The simplicity allows any modality to be represented uniformly
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional, Protocol, Tuple, runtime_checkable, Callable
from pathlib import Path


# =============================================================================
# PROTOCOL
# =============================================================================


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for datasets that can generate data for the manifold.
    
    The generate method yields (byte_value, sequence_index) tuples.
    This is the "universal tokenizer" format:
    
    - byte_value: Integer in [0, 255] representing a single byte
    - sequence_index: Position within the current logical unit
        - Resets to 0 at logical boundaries (new image, sentence, etc.)
        - This allows the system to learn positional structure
    """
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================


class SyntheticPattern(str, Enum):
    """Patterns for synthetic data generation."""
    
    # Random bytes
    RANDOM = "random"
    
    # Controlled collisions (shared bytes at specific positions)
    COLLISION = "collision"
    
    # Text patterns with shared prefixes
    TEXT_PREFIX = "text_prefix"
    
    # Repeated sequences
    REPEATED = "repeated"
    
    # Gradient (incrementing byte values)
    GRADIENT = "gradient"


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    # Pattern type
    pattern: SyntheticPattern = SyntheticPattern.RANDOM
    
    # Number of "files" or logical units
    num_units: int = 10
    
    # Length of each unit in bytes
    unit_length: int = 256
    
    # Random seed
    seed: int = 42
    
    # Pattern-specific parameters
    
    # For COLLISION pattern: fraction of positions that collide across units
    collision_rate: float = 0.5
    
    # For TEXT_PREFIX pattern: list of text patterns to use
    text_patterns: List[str] = field(default_factory=lambda: [
        "The cat sat.",
        "The cat ran.",
        "The dog sat.",
    ])
    
    # For TEXT_PREFIX pattern: repetitions per pattern
    pattern_counts: Optional[dict] = None
    
    # For REPEATED pattern: the byte sequence to repeat
    repeat_sequence: bytes = b"\x00\xff"


@dataclass
class FilesystemConfig:
    """Configuration for filesystem datasets."""
    
    # Path to file or directory
    path: Path = field(default_factory=lambda: Path("."))
    
    # Header bytes to skip (e.g., for MNIST IDX format)
    header_size: int = 0
    
    # Limit number of files to read (-1 = no limit)
    limit: int = -1
    
    # Offset: skip first N files
    offset: int = 0
    
    # For single files with segments (like MNIST)
    segment_size: int = 0  # 0 = no segmentation
    
    # File patterns to include (glob patterns)
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    
    # File patterns to exclude
    exclude_patterns: List[str] = field(default_factory=list)

    # Recurse into subdirectories (deterministic sorted walk)
    recursive: bool = False


@dataclass
class SpectralConfig:
    """Configuration for spectral (FFT-based) datasets."""
    
    # Source: path to audio file or directory
    path: Path = field(default_factory=lambda: Path("."))
    
    # FFT parameters
    fft_size: int = 1024
    hop_size: int = 256
    
    # Sample rate (for audio)
    sample_rate: int = 22050
    
    # Quantization: how many bits to use for magnitude (1-8)
    magnitude_bits: int = 8
    
    # Threshold: ignore bins below this fraction of max magnitude
    magnitude_threshold: float = 0.01
    
    # Whether to include phase information
    include_phase: bool = False


# =============================================================================
# BASE CLASS
# =============================================================================


class BaseDataset(ABC):
    """Base class for all datasets.
    
    Subclasses must implement generate() which yields (byte_value, sequence_index) tuples.
    This is the universal tokenizer format used throughout the system.
    """
    
    @abstractmethod
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples.
        
        This is the universal tokenizer format. Each tuple contains:
        - byte_value: Integer in [0, 255]
        - sequence_index: Position within current logical unit
        
        The sequence_index should reset to 0 at logical boundaries
        (e.g., new image, new sentence, new file).
        """
        raise NotImplementedError("Subclasses must implement generate()")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def bytes_to_tuples(
    data: bytes,
    start_index: int = 0,
) -> Iterator[Tuple[int, int]]:
    """Convert raw bytes to (byte_value, sequence_index) tuples.
    
    Args:
        data: Raw bytes to convert
        start_index: Starting sequence index (default 0)
    
    Yields:
        (byte_value, sequence_index) tuples
    """
    for i, byte in enumerate(data):
        yield (byte, start_index + i)


def stream_file(
    path: Path,
    header_size: int = 0,
    chunk_size: int = 4096,
) -> Iterator[Tuple[int, int]]:
    """Stream a file as (byte_value, sequence_index) tuples.
    
    Args:
        path: Path to file
        header_size: Bytes to skip at start
        chunk_size: Read buffer size
    
    Yields:
        (byte_value, sequence_index) tuples with index starting at 0
    """
    idx = 0
    with open(path, "rb") as f:
        if header_size > 0:
            f.read(header_size)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for byte in chunk:
                yield (byte, idx)
                idx += 1


def stream_files(
    paths: List[Path],
    header_size: int = 0,
    chunk_size: int = 4096,
) -> Iterator[Tuple[int, int]]:
    """Stream multiple files, resetting index at each file boundary.
    
    Args:
        paths: List of file paths
        header_size: Bytes to skip at start of each file
        chunk_size: Read buffer size
    
    Yields:
        (byte_value, sequence_index) tuples, index resets per file
    """
    for path in paths:
        yield from stream_file(path, header_size, chunk_size)


# =============================================================================
# SIMPLE WRAPPER
# =============================================================================


class BytesDataset:
    """Simple dataset wrapper for raw bytes.
    
    This is a convenience class that wraps raw bytes or a list of bytes
    into the DatasetProtocol format.
    
    Example:
        dataset = BytesDataset(b"Hello world")
        for byte_val, idx in dataset.generate():
            print(f"Byte {byte_val} at index {idx}")
    """
    
    def __init__(
        self,
        data: bytes | list[bytes],
        reset_per_item: bool = True,
    ):
        """Initialize the bytes dataset.
        
        Args:
            data: Raw bytes or list of bytes sequences
            reset_per_item: If True and data is a list, reset index per item
        """
        if isinstance(data, bytes):
            self.data = [data]
        else:
            self.data = list(data)
        self.reset_per_item = reset_per_item
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples."""
        global_idx = 0
        for item in self.data:
            local_idx = 0
            for byte in item:
                if self.reset_per_item:
                    yield (byte, local_idx)
                    local_idx += 1
                else:
                    yield (byte, global_idx)
                    global_idx += 1
