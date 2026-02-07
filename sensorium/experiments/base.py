from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        profile: bool = False,
        dashboard: bool = False,
        reportable: Sequence[str] | None = None,
    ):
        self.experiment_name = slugify(experiment_name)
        self.profile = profile
        self.dashboard = dashboard
        self.reportable = list(reportable) if reportable else []
