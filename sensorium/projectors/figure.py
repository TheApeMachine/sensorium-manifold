"""Figure projector for matplotlib charts.

Config-driven projector that generates figures from InferenceObserver data.

Example:
    projector = FigureProjector(FigureConfig(
        name="collision_metrics",
        chart_type="line",
        x="collision_rate",
        y=["compression_ratio", "entropy"],
        title="Collision Rate vs Metrics",
        xlabel="Collision Rate",
        ylabel="Value",
    ))
    
    projector.project(inference_observer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from sensorium.projectors.base import BaseProjector
from sensorium.console import console


if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class FigureConfig:
    """Configuration for figure generation.
    
    Attributes:
        name: Output filename (without extension)
        chart_type: Type of chart ("line", "scatter", "bar", "heatmap", "histogram")
        x: Field for x-axis (or None for index)
        y: Field(s) for y-axis (string or list of strings)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend (auto if multiple y fields)
        grid: Whether to show grid
        figsize: Figure size (width, height)
        dpi: Output DPI
        format: Output format ("png", "pdf", "svg")
        style: Matplotlib style ("default", "seaborn", etc.)
        colors: Optional list of colors for each y series
        markers: Optional list of markers for each y series
        xlim: Optional x-axis limits (min, max)
        ylim: Optional y-axis limits (min, max)
        log_x: Log scale for x-axis
        log_y: Log scale for y-axis
        annotate: Whether to annotate points with values
    """
    name: str = "figure"
    chart_type: str = "line"
    x: Optional[str] = None
    y: Union[str, List[str]] = field(default_factory=lambda: ["value"])
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    legend: Optional[bool] = None
    grid: bool = True
    figsize: tuple = (8, 6)
    dpi: int = 150
    format: str = "png"
    style: str = "default"
    colors: Optional[List[str]] = None
    markers: Optional[List[str]] = None
    xlim: Optional[tuple] = None
    ylim: Optional[tuple] = None
    log_x: bool = False
    log_y: bool = False
    annotate: bool = False


class FigureProjector(BaseProjector):
    """Generate matplotlib figures from InferenceObserver data."""
    
    def __init__(
        self,
        config: FigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Figure configuration
            output_dir: Output directory for figures
            **kwargs: Shortcut for config fields
        """
        console.info(f"Initializing figure projector with config: {config} and output directory: {output_dir}")
        super().__init__(output_dir or Path("paper/figures"))
        
        if config:
            self.config = config
        else:
            self.config = FigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate figure from source data.
        
        Args:
            source: InferenceObserver or dict with data
        
        Returns:
            Dict with status and output path
        """
        console.info(f"Projecting figure with source: {source}")
        import matplotlib.pyplot as plt
        import numpy as np
        
        console.info(f"Ensuring output directory exists: {self.output_dir}")
        self.ensure_output_dir()
        
        # Get results as list of dicts
        console.info(f"Getting results list from source: {source}")
        results = self._get_results_list(source)
        
        if not results:
            console.error("No results to project")
            return {"status": "error", "message": "No results to project"}
        
        # Apply style
        if self.config.style != "default":
            console.info(f"Using style: {self.config.style}")
            plt.style.use(self.config.style)
        
        # Create figure
        console.info(f"Creating figure with size: {self.config.figsize}")
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Normalize y to list
        console.info(f"Normalizing y to list: {self.config.y}")
        y_fields = self.config.y if isinstance(self.config.y, list) else [self.config.y]
        
        # Extract data
        console.info(f"Extracting data from results: {results} and field: {self.config.x}")
        x_data = self._extract_field(results, self.config.x)
        
        # Default colors and markers
        console.info(f"Default colors and markers: {default_colors} and {default_markers}")
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        default_markers = ["o", "s", "^", "D", "v"]
        
        console.info(f"Colors: {colors} and markers: {markers}")
        colors = self.config.colors or default_colors
        markers = self.config.markers or default_markers
        console.info(f"Colors: {colors} and markers: {markers}")
        
        # Plot each y field
        console.info(f"Plotting each y field: {y_fields}")
        for i, y_field in enumerate(y_fields):
            y_data = self._extract_field(results, y_field)
            console.info(f"Extracted y data: {y_data} for field: {y_field}")
            
            if not y_data:
                continue
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            label = y_field.replace("_", " ").title()
            
            if self.config.chart_type == "line":
                ax.plot(x_data, y_data, color=color, marker=marker, label=label)
            elif self.config.chart_type == "scatter":
                ax.scatter(x_data, y_data, color=color, marker=marker, label=label)
            elif self.config.chart_type == "bar":
                width = 0.8 / len(y_fields)
                offset = (i - len(y_fields) / 2 + 0.5) * width
                x_pos = np.arange(len(x_data)) + offset
                ax.bar(x_pos, y_data, width=width, color=color, label=label)
                ax.set_xticks(np.arange(len(x_data)))
                ax.set_xticklabels([str(x) for x in x_data])
            elif self.config.chart_type == "histogram":
                ax.hist(y_data, bins=20, color=color, alpha=0.7, label=label)
            
            # Annotate points
            if self.config.annotate and self.config.chart_type in ["line", "scatter"]:
                for x, y in zip(x_data, y_data):
                    if isinstance(y, (int, float)):
                        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                   xytext=(0, 5), ha="center", fontsize=8)
        
        # Configure axes
        if self.config.title:
            ax.set_title(self.config.title)
        if self.config.xlabel:
            ax.set_xlabel(self.config.xlabel)
        if self.config.ylabel:
            ax.set_ylabel(self.config.ylabel)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        # Axis scales
        if self.config.log_x:
            ax.set_xscale("log")
        if self.config.log_y:
            ax.set_yscale("log")
        
        # Axis limits
        if self.config.xlim:
            ax.set_xlim(self.config.xlim)
        if self.config.ylim:
            ax.set_ylim(self.config.ylim)
        
        # Legend
        show_legend = self.config.legend if self.config.legend is not None else len(y_fields) > 1
        if show_legend:
            ax.legend()
        
        # Save
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        console.info(f"Saving figure to: {output_path}")
        fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return {
            "status": "success",
            "path": str(output_path),
            "chart_type": self.config.chart_type,
            "series": len(y_fields),
        }
    
    def _extract_field(self, results: List[Dict], field: Optional[str]) -> List[Any]:
        """Extract a field from results, or generate index if None."""
        if field is None:
            return list(range(len(results)))
        return [r.get(field) for r in results]
