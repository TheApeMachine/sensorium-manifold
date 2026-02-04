"""Base class for all datasets"""

from abc import ABC, abstractmethod
from typing import Iterator, List
from pathlib import Path
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Configuration for all datasets"""
    name: str = "filesystem"
    num_classes: int = 0  # Not needed for filesystem datasets
    num_examples_per_class: int = 0  # Not needed for filesystem datasets
    segment_size: int = 0  # Not needed for filesystem datasets
    path: Path
    # Optional fields with defaults (not needed for filesystem datasets)
    seed: int = 42
    vocab: List[bytes] = None  # type: ignore
    header_size: int = 0  # For binary file formats with headers
    limit: int = -1  # For limiting the number of examples
    offset: int = -1  # For skipping the first n examples
    
    class Config:
        arbitrary_types_allowed = True


class BaseDataset(ABC):
    """Base class for all datasets"""
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def generate(self) -> Iterator[bytes]:
        """Implement a generator that yields (chunks of) bytes"""
        raise NotImplementedError("Subclasses must implement this method")