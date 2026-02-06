"""Prototype tokenizer for the Sensorium Manifold.

This tokenizer is a prototype for the Universal Tokenizer.
It is a simple tokenizer that tokenizes the data into tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, TypedDict

class Tokenizer:
    """Prototype tokenizer for the Sensorium Manifold.
    
    This tokenizer is a prototype for the Universal Tokenizer.
    It is a simple tokenizer that tokenizes the data into tokens.
    """

    def tokenize(self, data: bytes) -> torch.Tensor:
        """Tokenize the data into tokens."""
        raise NotImplementedError("PrototypeTokenizer.tokenize is not implemented")

    def stream(self) -> Iterator[TokenBatch]:
        """Stream the tokens from the data."""
        raise NotImplementedError("PrototypeTokenizer.stream is not implemented")