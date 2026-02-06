"""Synthetic dataset for generating controlled byte patterns.

This dataset generates synthetic data that is useful for testing and 
validating the thermodynamic manifold's behavior under controlled conditions.

Patterns:
- RANDOM: Pure random bytes (baseline/noise)
- COLLISION: Controlled hash collisions (shared bytes at positions)
- TEXT_PREFIX: Text patterns sharing prefixes (for trie testing)
- REPEATED: Repeating byte sequences
- GRADIENT: Incrementing byte values
"""

from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np

from sensorium.dataset.base import BaseDataset, SyntheticConfig, SyntheticPattern


class SyntheticDataset(BaseDataset):
    """Synthetic dataset for generating controlled byte patterns.
    
    Supports multiple pattern types for different testing scenarios:
    
    - RANDOM: Pure random bytes per unit
    - COLLISION: Shared bytes at specific positions across units  
    - TEXT_PREFIX: Text patterns with shared prefixes
    - REPEATED: Repeating byte sequences
    - GRADIENT: Incrementing byte values
    
    Examples:
        # Random bytes
        dataset = SyntheticDataset(SyntheticConfig(
            pattern=SyntheticPattern.RANDOM,
            num_units=10,
            unit_length=256,
        ))
        
        # Controlled collisions (50% of positions shared)
        dataset = SyntheticDataset(SyntheticConfig(
            pattern=SyntheticPattern.COLLISION,
            num_units=20,
            unit_length=32,
            collision_rate=0.5,
        ))
        
        # Text patterns with prefixes
        dataset = SyntheticDataset(SyntheticConfig(
            pattern=SyntheticPattern.TEXT_PREFIX,
            text_patterns=["The cat sat.", "The cat ran.", "The dog sat."],
            pattern_counts={"The cat sat.": 30, "The cat ran.": 20, "The dog sat.": 10},
        ))
    """
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)
        
        # Pre-generate pattern data based on type
        self._patterns: list[bytes] = []
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize patterns based on config."""
        pattern = self.config.pattern
        
        if pattern == SyntheticPattern.RANDOM:
            self._init_random()
        elif pattern == SyntheticPattern.COLLISION:
            self._init_collision()
        elif pattern == SyntheticPattern.TEXT_PREFIX:
            self._init_text_prefix()
        elif pattern == SyntheticPattern.REPEATED:
            self._init_repeated()
        elif pattern == SyntheticPattern.GRADIENT:
            self._init_gradient()
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def _init_random(self):
        """Generate random byte patterns."""
        num_units = int(self.config.num_units)
        unit_length = int(self.config.unit_length)
        if num_units <= 0 or unit_length <= 0:
            self._patterns = []
            return

        # Vectorized generation (much faster at large scale).
        data = self._rng.randint(0, 256, size=(num_units, unit_length), dtype=np.uint8)
        self._patterns.extend(row.tobytes() for row in data)
    
    def _init_collision(self):
        """Generate patterns with controlled collisions.
        
        At `collision_rate` fraction of positions, all units share the same byte.
        At other positions, each unit has a unique random byte.
        """
        unit_length = int(self.config.unit_length)
        num_units = int(self.config.num_units)
        if num_units <= 0 or unit_length <= 0:
            self._patterns = []
            return
        
        # Generate base pattern (shared across all units at collision positions)
        base_pattern = self._rng.randint(0, 256, size=unit_length, dtype=np.uint8)
        
        # Generate unique patterns for each unit
        unique_patterns = self._rng.randint(0, 256, size=(num_units, unit_length), dtype=np.uint8)
        
        # Determine collision positions
        num_collisions = int(unit_length * self.config.collision_rate)
        collision_positions_arr = self._rng.choice(unit_length, size=num_collisions, replace=False)
        collision_positions = set(int(x) for x in collision_positions_arr.tolist())
        collision_mask = np.zeros((unit_length,), dtype=bool)
        collision_mask[list(collision_positions)] = True
        
        # Build patterns (vectorized: apply base bytes at collision positions)
        out = unique_patterns.copy()
        out[:, collision_mask] = base_pattern[collision_mask][None, :]
        self._patterns.extend(row.tobytes() for row in out)
    
    def _init_text_prefix(self):
        """Generate text patterns, optionally with frequency weighting."""
        text_patterns = self.config.text_patterns
        pattern_counts = self.config.pattern_counts
        
        if pattern_counts:
            # Use explicit counts
            for pattern, count in pattern_counts.items():
                for _ in range(count):
                    self._patterns.append(pattern.encode("utf-8"))
        else:
            # Use num_units: repeat patterns evenly
            reps_per_pattern = max(1, self.config.num_units // len(text_patterns))
            for pattern in text_patterns:
                for _ in range(reps_per_pattern):
                    self._patterns.append(pattern.encode("utf-8"))
        
        # Shuffle patterns
        self._rng.shuffle(self._patterns)
    
    def _init_repeated(self):
        """Generate patterns from repeating byte sequences."""
        repeat_seq = self.config.repeat_sequence
        unit_length = self.config.unit_length
        
        for _ in range(self.config.num_units):
            # Repeat the sequence to fill unit_length
            repeats = (unit_length // len(repeat_seq)) + 1
            data = (repeat_seq * repeats)[:unit_length]
            self._patterns.append(data)
    
    def _init_gradient(self):
        """Generate gradient patterns (incrementing byte values)."""
        unit_length = self.config.unit_length
        
        for i in range(self.config.num_units):
            # Each unit starts at a different offset
            start = (i * 17) % 256  # Prime step for variety
            data = bytes((start + j) % 256 for j in range(unit_length))
            self._patterns.append(data)
    
    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate (byte_value, sequence_index) tuples.
        
        Each pattern is a logical unit; the sequence index resets to 0
        at each pattern boundary.
        """
        for pattern in self._patterns:
            for idx, byte_val in enumerate(pattern):
                yield (byte_val, idx)
    
    # =========================================================================
    # Convenience methods
    # =========================================================================
    
    def get_patterns(self) -> list[bytes]:
        """Return the generated patterns for inspection."""
        return list(self._patterns)
    
    def get_collision_positions(self) -> set[int] | None:
        """Return collision positions if COLLISION pattern was used."""
        if self.config.pattern != SyntheticPattern.COLLISION:
            return None
        
        # Recompute (deterministic due to seed)
        unit_length = int(self.config.unit_length)
        num_collisions = int(unit_length * self.config.collision_rate)
        rng = np.random.RandomState(self.config.seed)
        rng.randint(0, 256, size=unit_length)  # base pattern
        rng.randint(0, 256, size=(int(self.config.num_units), unit_length))  # unique patterns
        return set(int(x) for x in rng.choice(unit_length, size=num_collisions, replace=False).tolist())
