"""Aggregate observers that reduce observations to summary values.

These observers take observation results and compute aggregate
statistics like counts, sums, and statistical measures.
"""

from .count import Count
from .total import Total
from .mean import Mean
from .statistics import Statistics

__all__ = [
    "Count",
    "Total",
    "Mean",
    "Statistics",
]
