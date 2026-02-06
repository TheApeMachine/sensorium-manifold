"""Filesystem dataset for loading raw bytes from files.

This dataset reads files from the filesystem and yields their bytes
as (byte_value, sequence_index) tuples. It handles:

- Single files
- Directories (iterates all matching files)
- Segmented files (e.g., MNIST IDX format with fixed-size images)
- Headers to skip
- Include/exclude patterns
"""

from pathlib import Path
from fnmatch import fnmatch
from typing import Iterator, Tuple

from sensorium.dataset.base import BaseDataset, FilesystemConfig


class FilesystemDataset(BaseDataset):
    """Dataset that loads raw bytes from files.
    
    Implements the DatasetProtocol, yielding (byte_value, sequence_index) tuples.
    The sequence index resets at file/segment boundaries.
    
    Examples:
        # Load all files from a directory
        dataset = FilesystemDataset(FilesystemConfig(
            path=Path("./data/texts"),
        ))
        
        # Load MNIST images (skip header, read segments)
        dataset = FilesystemDataset(FilesystemConfig(
            path=Path("./data/mnist/train-images-idx3-ubyte"),
            header_size=16,  # MNIST header
            segment_size=784,  # 28x28 pixels
            limit=1000,  # First 1000 images
        ))
        
        # Load specific file types
        dataset = FilesystemDataset(FilesystemConfig(
            path=Path("./data/audio"),
            include_patterns=["*.wav", "*.mp3"],
            exclude_patterns=["*_temp*"],
        ))
    """
    
    def __init__(self, config: FilesystemConfig):
        self.config = config
    
    def _matches_patterns(self, path: Path) -> bool:
        """Check if path matches include patterns and not exclude patterns."""
        name = path.name
        
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if fnmatch(name, pattern):
                return False
        
        # Check include patterns
        for pattern in self.config.include_patterns:
            if fnmatch(name, pattern):
                return True
        
        # Default: include if no patterns specified or didn't match any include
        return len(self.config.include_patterns) == 0 or "*" in self.config.include_patterns
    
    def _get_files(self) -> list[Path]:
        """Get list of files to process."""
        path = self.config.path
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        if path.is_file():
            return [path]
        
        # Directory: get matching files (optionally recursive).
        #
        # [CHOICE] skip build/cache dirs by default
        # [REASON] prevents accidental ingestion of LaTeX build artifacts, bytecode,
        #          and other generated noise when running corpus experiments.
        _SKIP_DIRNAMES = {"build", "__pycache__", ".git", ".venv", ".mypy_cache", ".pytest_cache"}

        if bool(getattr(self.config, "recursive", False)):
            cand = (p for p in sorted(path.rglob("*")) if p.is_file())
            files = []
            for p in cand:
                if any(part in _SKIP_DIRNAMES for part in p.parts):
                    continue
                if self._matches_patterns(p):
                    files.append(p)
        else:
            files = [
                p for p in sorted(path.iterdir())
                if p.is_file() and self._matches_patterns(p)
            ]
        
        if not files:
            raise ValueError(f"No matching files found at: {path}")
        
        # Apply offset and limit
        start = self.config.offset
        end = start + self.config.limit if self.config.limit > 0 else None
        return files[start:end]
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Load files and yield (byte_value, sequence_index) tuples.
        
        The sequence index resets to 0 at each file/segment boundary.
        """
        files = self._get_files()
        
        if len(files) == 1 and self.config.segment_size > 0:
            # Single file with segments: yield segments as logical units
            yield from self._generate_segmented(files[0])
        else:
            # Multiple files: each file is a logical unit
            yield from self._generate_files(files)
    
    def _generate_files(self, files: list[Path]) -> Iterator[Tuple[int, int]]:
        """Yield bytes from multiple files, resetting index per file."""
        for path in files:
            data = path.read_bytes()
            
            # Skip header if specified
            if self.config.header_size > 0:
                data = data[self.config.header_size:]
            
            # Yield (byte, index) tuples - index resets per file
            for idx, byte in enumerate(data):
                yield (byte, idx)
    
    def _generate_segmented(self, path: Path) -> Iterator[Tuple[int, int]]:
        """Yield bytes from a segmented file (e.g., MNIST).
        
        Each segment is a logical unit; index resets per segment.
        """
        data = path.read_bytes()
        
        # Skip header
        if self.config.header_size > 0:
            data = data[self.config.header_size:]
        
        segment_size = self.config.segment_size
        
        # Apply offset and limit (in terms of segments)
        start_segment = self.config.offset if self.config.offset > 0 else 0
        end_segment = start_segment + self.config.limit if self.config.limit > 0 else len(data) // segment_size
        
        start_byte = start_segment * segment_size
        end_byte = min(end_segment * segment_size, len(data))
        
        # Yield (byte, index) tuples - index resets per segment
        for i in range(start_byte, end_byte, segment_size):
            segment = data[i:i + segment_size]
            for idx, byte in enumerate(segment):
                yield (byte, idx)
