"""Rule-shift dataset for online adaptation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence, Tuple

from sensorium.dataset.base import BaseDataset


@dataclass
class RuleShiftPhase:
    """One phase in a rule-shift curriculum."""

    name: str
    phrase: str
    reps: int


@dataclass
class RuleShiftConfig:
    """Configuration for a rule-shift curriculum."""

    phases: Sequence[RuleShiftPhase] = field(
        default_factory=lambda: (
            RuleShiftPhase(name="forward", phrase="The cat sat on the mat.", reps=50),
            RuleShiftPhase(name="reverse", phrase="mat the on sat cat The.", reps=50),
        )
    )
    segment_size: int = 24


class RuleShiftDataset(BaseDataset):
    """Dataset for curriculum-style rule-shift experiments."""

    def __init__(self, config: RuleShiftConfig | None = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = RuleShiftConfig(**kwargs)

        self.phases = [
            RuleShiftPhase(
                name=str(phase.name),
                phrase=str(phase.phrase).ljust(self.config.segment_size)[
                    : self.config.segment_size
                ],
                reps=int(max(1, phase.reps)),
            )
            for phase in self.config.phases
        ]
        if not self.phases:
            raise ValueError("RuleShiftConfig.phases must contain at least one phase")

        blocks = [phase.phrase * phase.reps for phase in self.phases]
        self.full_text = "".join(blocks)
        self.train_bytes = self.full_text.encode("utf-8")

        self.phase_schedule: list[dict[str, int | str]] = []
        rep_cursor = 0
        byte_cursor = 0
        for phase in self.phases:
            start_rep = rep_cursor
            end_rep = rep_cursor + int(phase.reps)
            start_byte = byte_cursor
            end_byte = byte_cursor + int(phase.reps) * int(self.config.segment_size)
            self.phase_schedule.append(
                {
                    "name": phase.name,
                    "phrase": phase.phrase,
                    "start_rep": int(start_rep),
                    "end_rep": int(end_rep),
                    "start_byte": int(start_byte),
                    "end_byte": int(end_byte),
                }
            )
            rep_cursor = end_rep
            byte_cursor = end_byte

        # Compatibility aliases expected by existing callers.
        self.forward_phrase = str(self.phase_schedule[0]["phrase"])
        self.reverse_phrase = str(self.phase_schedule[-1]["phrase"])
        self.phase_switch_byte = int(
            self.phase_schedule[1]["start_byte"]
            if len(self.phase_schedule) > 1
            else len(self.train_bytes)
        )

    @property
    def segment_size(self) -> int:
        """Return segment size for external access."""
        return self.config.segment_size

    @property
    def forward_reps(self) -> int:
        """Compatibility: repetitions of first phase."""
        return int(self.phases[0].reps)

    @property
    def reverse_reps(self) -> int:
        """Compatibility: repetitions of last phase."""
        return int(self.phases[-1].reps)

    @property
    def total_reps(self) -> int:
        """Total repetitions across all phases."""
        return int(sum(int(phase.reps) for phase in self.phases))

    def generate(self) -> Iterator[Tuple[int, int]]:
        """Generate training data as (byte_value, sequence_index) tuples."""
        for idx, byte_val in enumerate(self.train_bytes):
            yield (byte_val, idx)

    def __repr__(self) -> str:
        names = " -> ".join(phase.name for phase in self.phases)
        return (
            f"RuleShiftDataset(phases='{names}', segment_size={self.config.segment_size}, "
            f"total_reps={self.total_reps})"
        )
