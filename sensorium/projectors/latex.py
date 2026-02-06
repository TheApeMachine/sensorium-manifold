"""LaTeX table projector.

Config-driven projector that generates LaTeX tables from InferenceObserver data.

Example:
    projector = LaTeXTableProjector(TableConfig(
        name="collision_summary",
        columns=["collision_rate", "compression_ratio", "entropy"],
        headers={"collision_rate": "Rate", "compression_ratio": "Comp."},  # Optional
        caption="Collision rate vs compression metrics",
        label="tab:collision",
        precision=3,
    ))
    
    projector.project(inference_observer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from sensorium.projectors.base import BaseProjector

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class TableConfig:
    """Configuration for LaTeX table generation.
    
    Attributes:
        name: Output filename (without .tex extension)
        columns: List of field names to include as columns
        headers: Optional dict mapping field names to display headers
        caption: Table caption
        label: LaTeX label for referencing
        precision: Decimal precision for floats
        alignment: Column alignment ("l", "c", "r" or per-column string)
        bold_best: If True, bold the best value in each numeric column
        sort_by: Optional field to sort rows by
        sort_reverse: If True, sort descending
    """
    name: str = "table"
    columns: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    caption: str = ""
    label: str = ""
    precision: int = 3
    alignment: str = "c"
    bold_best: bool = False
    sort_by: Optional[str] = None
    sort_reverse: bool = False


class LaTeXTableProjector(BaseProjector):
    """Generate LaTeX tables from InferenceObserver data."""
    
    def __init__(
        self,
        config: TableConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Table configuration
            output_dir: Output directory for .tex files
            **kwargs: Shortcut for config fields (name, columns, etc.)
        """
        super().__init__(output_dir or Path("paper/tables"))
        
        # Allow config or kwargs
        if config:
            self.config = config
        else:
            self.config = TableConfig(**kwargs)

        self.headers = self.config.headers or [
            col.replace("_", " ").title() for col in self.config.columns
        ]
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LaTeX table from source data.
        
        Args:
            source: InferenceObserver or dict with data
        
        Returns:
            Dict with status and output path
        """
        self.ensure_output_dir()
        
        # Get results as list of dicts
        results = self._get_results_list(source)
        
        if not results:
            return {"status": "error", "message": "No results to project"}
        
        # Determine columns
        columns = self.config.columns if self.config.columns else list(results[0].keys())
        
        # Sort if specified
        if self.config.sort_by and self.config.sort_by in columns:
            results = sorted(
                results,
                key=lambda r: r.get(self.config.sort_by, 0),
                reverse=self.config.sort_reverse,
            )
        
        # Generate LaTeX
        latex = self._generate_latex(results, columns)
        
        # Write to file
        output_path = self.output_dir / f"{self.config.name}.tex"
        output_path.write_text(latex)
        
        return {
            "status": "success",
            "path": str(output_path),
            "rows": len(results),
            "columns": len(columns),
        }
    
    def _generate_latex(self, results: List[Dict], columns: List[str]) -> str:
        """Generate LaTeX table content."""
        lines = []
        
        # Get headers
        headers = [self.config.headers.get(c, c.replace("_", " ").title()) for c in columns]
        
        # Determine alignment
        if len(self.config.alignment) == 1:
            align = self.config.alignment * len(columns)
        else:
            align = self.config.alignment
        
        # Table header
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        if self.config.caption:
            lines.append(rf"\caption{{{self.config.caption}}}")
        if self.config.label:
            lines.append(rf"\label{{{self.config.label}}}")
        lines.append(rf"\begin{{tabular}}{{{align}}}")
        lines.append(r"\toprule")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")
        
        # Find best values for each column (if bold_best)
        best_values = {}
        if self.config.bold_best:
            for c in columns:
                values = [r.get(c) for r in results if isinstance(r.get(c), (int, float))]
                if values:
                    best_values[c] = max(values)
        
        # Data rows
        for r in results:
            cells = []
            for c in columns:
                val = r.get(c, "")
                formatted = self._format_value(val)
                
                # Bold if best
                if self.config.bold_best and c in best_values:
                    if val == best_values[c]:
                        formatted = rf"\textbf{{{formatted}}}"
                
                cells.append(formatted)
            lines.append(" & ".join(cells) + r" \\")
        
        # Table footer
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    def _format_value(self, val: Any) -> str:
        """Format a value for LaTeX."""
        if val is None:
            return "—"
        elif isinstance(val, bool):
            return r"\checkmark" if val else "—"
        elif isinstance(val, float):
            if abs(val) < 0.001 and val != 0:
                return f"${val:.2e}$"
            return f"${val:.{self.config.precision}f}$"
        elif isinstance(val, int):
            return str(val)
        else:
            # Escape LaTeX special characters
            s = str(val)
            for char in ["_", "%", "&", "#", "$"]:
                s = s.replace(char, f"\\{char}")
            return s
