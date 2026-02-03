from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LaTeXTable:
    """Container for LaTeX table outputs."""
    name: str
    header: List[str]
    rows: List[List[str|int|float]]


    def to_latex(self) -> str:
        """Convert the table to LaTeX format."""
        return f"\\begin{{table}}[h]\n\\centering\n\\caption{{{self.name}}}\n\\label{{tab:{self.name}}}\n\\begin{{tabular}}{{{'l' * len(self.header)}}}\n\\toprule\n{' & '.join(self.header)}\n\\midrule\n{'\n'.join([' & '.join(map(str, row)) for row in self.rows])}n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"

    def to_markdown(self) -> str:
        """Convert the table to Markdown format."""
        return f"| {self.name} |\n| {' | '.join(self.header)} |\n| {' | '.join([' | '.join(map(str, row)) for row in self.rows])} |\n"


@dataclass
class ExperimentResult:
    """Container for experiment outputs."""
    name: str
    goal: str
    metrics: Dict[str, float|int|str|List[float|int|str]]
    tables: Dict[str, str]  # name -> LaTeX content
    figures: Dict[str, Path]  # name -> path to generated figure