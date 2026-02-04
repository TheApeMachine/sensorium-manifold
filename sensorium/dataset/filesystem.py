from pathlib import Path
import numpy as np
from typing import Iterator

from sensorium.dataset.base import BaseDataset, DatasetConfig


class FilesystemDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def generate(self) -> Iterator[bytes]:
        """Load files from the filesystem and yield as bytes chunks."""

        if not self.config.path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.config.path}"
            )
        
        # Handle both files and directories
        if self.config.path.is_file():
            files = [self.config.path]
        else:
            files = [
                p for p in sorted(self.config.path.iterdir()) if p.is_file()
            ]
        
        if not files:
            raise ValueError(f"No files found at: {self.config.path}")
        
        # For single files, limit/offset apply to bytes (or segments if segment_size is set)
        # For multiple files, limit/offset apply to number of files
        if len(files) == 1 and self.config.segment_size > 0:
            # Single file with segment_size: limit/offset are in terms of segments
            path = files[0]
            all_data = path.read_bytes()[self.config.header_size:]
            
            segment_size = self.config.segment_size
            offset_bytes = (self.config.offset if self.config.offset > 0 else 0) * segment_size
            limit_bytes = (self.config.limit if self.config.limit > 0 else len(all_data)) * segment_size
            
            start = offset_bytes
            end = start + limit_bytes if limit_bytes > 0 else len(all_data)
            data = all_data[start:end]
            
            print(f"Loaded {path.name}: {len(data):,} bytes ({len(data) // segment_size} segments)")
            yield data
        else:
            # Multiple files or no segment_size: limit/offset apply to number of files
            for path in files[
                self.config.offset:self.config.offset + self.config.limit
                if self.config.limit > 0 else None
            ]:
                data = path.read_bytes()[self.config.header_size:]
                print(f"Loaded {path.name}: {len(data):,} bytes")
                yield data