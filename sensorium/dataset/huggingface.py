"""HuggingFace dataset wrapper.

Streams datasets from Hugging Face Hub and yields (byte_value, sequence_index) tuples.

Example:
    from sensorium.dataset import HuggingFaceDataset, HuggingFaceConfig
    
    # Stream FineWeb-Edu
    dataset = HuggingFaceDataset(HuggingFaceConfig(
        name="HuggingFaceFW/fineweb-edu",
        split="train",
        field="text",
        streaming=True,
    ))
    
    # Use in manifold
    manifold.add_dataset(dataset.generate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

from sensorium.dataset.base import BaseDataset


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace dataset loading.
    
    Attributes:
        name: Dataset name on HF Hub (e.g., "HuggingFaceFW/fineweb-edu")
        split: Dataset split ("train", "test", "validation")
        field: Field to extract text from (e.g., "text", "content")
        streaming: If True, stream data without downloading full dataset
        subset: Optional dataset subset/configuration name
        max_samples: Maximum samples to load (None for unlimited)
        encoding: Text encoding for byte conversion
        trust_remote_code: Whether to trust remote code in dataset scripts
        revision: Optional specific revision/commit
    """
    name: str = ""
    split: str = "train"
    field: str = "text"
    streaming: bool = True
    subset: Optional[str] = None
    max_samples: Optional[int] = None
    encoding: str = "utf-8"
    trust_remote_code: bool = False
    revision: Optional[str] = None


class HuggingFaceDataset(BaseDataset):
    """Stream datasets from Hugging Face Hub.
    
    Wraps the datasets library to provide seamless integration with the
    Sensorium tokenizer. Supports streaming for large datasets like FineWeb-Edu.
    """
    
    def __init__(self, config: HuggingFaceConfig):
        """
        Args:
            config: Dataset configuration
        """
        self.config = config
        self._dataset = None
    
    def _load_dataset(self) -> Any:
        """Lazy-load the dataset."""
        if self._dataset is not None:
            return self._dataset
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required for HuggingFaceDataset. "
                "Install with: pip install datasets"
            )
        
        kwargs: Dict[str, Any] = {
            "streaming": self.config.streaming,
            "split": self.config.split,
            "trust_remote_code": self.config.trust_remote_code,
        }
        
        if self.config.subset:
            kwargs["name"] = self.config.subset
        
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        
        self._dataset = load_dataset(self.config.name, **kwargs)
        return self._dataset
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples from dataset.
        
        Each sample in the dataset is treated as a separate logical unit,
        with sequence_index resetting at sample boundaries.
        
        Yields:
            Tuples of (byte_value, sequence_index)
        """
        dataset = self._load_dataset()
        
        sample_count = 0
        for sample in dataset:
            # Check max samples
            if self.config.max_samples and sample_count >= self.config.max_samples:
                break
            
            # Extract text field
            text = sample.get(self.config.field, "")
            if not text:
                continue
            
            # Convert to bytes and yield with sequence index
            try:
                text_bytes = text.encode(self.config.encoding)
            except (UnicodeEncodeError, AttributeError):
                continue
            
            for idx, byte_val in enumerate(text_bytes):
                yield (byte_val, idx)
            
            sample_count += 1
    
    def __repr__(self) -> str:
        return (
            f"HuggingFaceDataset(name={self.config.name!r}, "
            f"split={self.config.split!r}, field={self.config.field!r})"
        )


# Convenience factory functions

def fineweb_edu(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> HuggingFaceDataset:
    """Create a FineWeb-Edu dataset.
    
    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        streaming: Stream data without downloading
    
    Returns:
        HuggingFaceDataset configured for FineWeb-Edu
    """
    return HuggingFaceDataset(HuggingFaceConfig(
        name="HuggingFaceFW/fineweb-edu",
        split=split,
        field="text",
        streaming=streaming,
        max_samples=max_samples,
    ))


def the_pile(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> HuggingFaceDataset:
    """Create a The Pile dataset.
    
    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        streaming: Stream data without downloading
    
    Returns:
        HuggingFaceDataset configured for The Pile
    """
    return HuggingFaceDataset(HuggingFaceConfig(
        name="EleutherAI/pile",
        split=split,
        field="text",
        streaming=streaming,
        max_samples=max_samples,
    ))


def wikipedia(
    language: str = "en",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> HuggingFaceDataset:
    """Create a Wikipedia dataset.
    
    Args:
        language: Wikipedia language code
        split: Dataset split
        max_samples: Maximum samples to load
        streaming: Stream data without downloading
    
    Returns:
        HuggingFaceDataset configured for Wikipedia
    """
    return HuggingFaceDataset(HuggingFaceConfig(
        name="wikipedia",
        subset=f"20220301.{language}",
        split=split,
        field="text",
        streaming=streaming,
        max_samples=max_samples,
    ))
