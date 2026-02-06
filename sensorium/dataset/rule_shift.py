"""Rule shift dataset for online adaptation experiments.

This dataset generates forward-then-reverse phrase patterns to test
how the manifold adapts when patterns suddenly change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

from sensorium.dataset.base import BaseDataset


@dataclass
class RuleShiftConfig:
    """Configuration for rule shift dataset."""
    forward_phrase: str = "The cat sat on the mat."
    reverse_phrase: str = "mat the on sat cat The."
    forward_reps: int = 50
    reverse_reps: int = 50
    segment_size: int = 24


class RuleShiftDataset(BaseDataset):
    """Dataset for rule shift experiment with forward then reverse phrases.
    
    Creates a sequence that first repeats a forward phrase, then switches
    to a reversed version. Used to test online adaptation.
    """
    
    def __init__(self, config: RuleShiftConfig | None = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = RuleShiftConfig(**kwargs)
        
        # Pad phrases to segment size
        self.forward_phrase = self.config.forward_phrase.ljust(self.config.segment_size)[:self.config.segment_size]
        self.reverse_phrase = self.config.reverse_phrase.ljust(self.config.segment_size)[:self.config.segment_size]
        
        # Build data
        forward_text = self.forward_phrase * self.config.forward_reps
        reverse_text = self.reverse_phrase * self.config.reverse_reps
        self.full_text = forward_text + reverse_text
        self.train_bytes = self.full_text.encode("utf-8")
        self.phase_switch_byte = len(forward_text)
    
    @property
    def segment_size(self) -> int:
        """Return segment size for external access."""
        return self.config.segment_size
    
    @property
    def forward_reps(self) -> int:
        """Return forward repetitions for external access."""
        return self.config.forward_reps
    
    @property
    def reverse_reps(self) -> int:
        """Return reverse repetitions for external access."""
        return self.config.reverse_reps
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples."""
        for idx, byte_val in enumerate(self.train_bytes):
            yield (byte_val, idx)
    
    def __repr__(self) -> str:
        return (f"RuleShiftDataset(forward='{self.forward_phrase[:20]}...', "
                f"reverse='{self.reverse_phrase[:20]}...', "
                f"forward_reps={self.config.forward_reps}, "
                f"reverse_reps={self.config.reverse_reps})")
