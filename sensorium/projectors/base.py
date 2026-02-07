"""Base projector protocol and implementations.

Projectors transform observation results into output artifacts.
They query the InferenceObserver directly for data fields.

Design:
- Projectors take an InferenceObserver as input to project()
- They query the observer for specific fields by name
- Configuration specifies what fields to extract and how to format

Example:
    # Projector queries inference observer
    projector = LaTeXTableProjector(TableConfig(
        columns=["collision_rate", "compression_ratio", "entropy"],
        name="summary",
    ))
    
    # project() receives the inference observer, not a pre-extracted dict
    projector.project(inference_observer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol, Union, TYPE_CHECKING

from sensorium.console import console


if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


class ProjectorProtocol(Protocol):
    """Protocol for projectors that generate output artifacts."""
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Project observation data to output artifact.
        
        Args:
            source: InferenceObserver to query, or a dict
        
        Returns:
            Dict with status and output details
        """
        ...


class BaseProjector(ABC):
    """Base class for projectors with common functionality."""
    
    def __init__(self, output_dir: Path | None = None):
        console.info(f"Initializing projector with output directory: {output_dir}")
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts")
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        console.info(f"Ensuring output directory exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_data(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data from source (InferenceObserver or dict).
        
        This provides a unified interface for projectors to get data.
        """
        console.info(f"Extracting data from source: {source}")
        # If it's an InferenceObserver, convert to column-oriented dict
        if hasattr(source, "as_dict"):
            return source.as_dict()
        elif hasattr(source, "results"):
            # Has results list, convert to dict
            return {
                "results": source.results,
                **{k: [r.get(k) for r in source.results] for k in (source.results[0].keys() if source.results else [])}
            }
        elif isinstance(source, dict):
            return source
        else:
            return {}
    
    def _get_results_list(self, source: Union["InferenceObserver", Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get results as a list of dicts (row-oriented)."""
        console.info(f"Getting results list from source: {source}")
        if hasattr(source, "results"):
            return source.results
        elif isinstance(source, dict):
            # Check if it already has a results list
            if "results" in source and isinstance(source["results"], list):
                return source["results"]
            # Otherwise treat the whole dict as a single result
            return [source]
        else:
            return []
    
    @abstractmethod
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Project data to output artifact."""
        pass


class PipelineProjector(BaseProjector):
    """Compose multiple projectors into a pipeline."""
    
    def __init__(self, *projectors: ProjectorProtocol, output_dir: Path | None = None):
        console.info(f"Initializing pipeline projector with output directory: {output_dir}")
        super().__init__(output_dir)
        self.projectors = list(projectors)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Run all projectors and collect results."""
        console.info(f"Running pipeline projector with source: {source}")
        results: Dict[str, Any] = {}
        console.info(f"Projectors: {self.projectors}")
        
        for projector in self.projectors:
            name = projector.__class__.__name__
            console.info(f"Running projector: {name}")
            try:
                result = projector.project(source)
                results[name] = result
            except Exception as e:
                console.error(f"Error running projector: {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def add(self, projector: ProjectorProtocol) -> "PipelineProjector":
        """Add a projector to the pipeline. Returns self for chaining."""
        self.projectors.append(projector)
        return self


class ConsoleProjector(BaseProjector):
    """Projector that prints results to console."""
    
    def __init__(self, fields: List[str] | None = None, format: str = "table"):
        """
        Args:
            fields: Fields to print (None = all)
            format: Output format ("table", "json", "simple")
        """
        console.info(f"Initializing console projector with fields: {fields} and format: {format}")
        super().__init__()
        self.fields = fields
        self.format = format
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Print results to console."""
        console.info(f"Projecting console with source: {source}")
        results = self._get_results_list(source)
        
        if not results:
            console.error("No results to display.")
            return {"status": "empty"}
        
        # Determine fields to show
        if self.fields:
            fields = self.fields
        else:
            fields = list(results[0].keys())
        
        if self.format == "table":
            console.info(f"Printing table with results: {results} and fields: {fields}")
            self._print_table(results, fields)
        elif self.format == "json":
            import json
            console.info(f"Printing JSON with results: {results}")
            console.info(json.dumps(results, indent=2, default=str))
        else:
            for i, r in enumerate(results):
                console.info(f"\n--- Result {i+1} ---")
                for f in fields:
                    if f in r:
                        console.info(f"  {f}: {r[f]}")
        
        return {"status": "success", "count": len(results)}
    
    def _print_table(self, results: List[Dict], fields: List[str]):
        """Print results as a simple table."""
        # Calculate column widths
        widths = {f: len(f) for f in fields}
        for r in results:
            for f in fields:
                val = str(r.get(f, ""))[:20]  # Truncate
                widths[f] = max(widths[f], len(val))
        
        # Print header
        header = " | ".join(f.ljust(widths[f]) for f in fields)
        console.info(header)
        print("-" * len(header))
        
        # Print rows
        for r in results:
            row = " | ".join(str(r.get(f, ""))[:20].ljust(widths[f]) for f in fields)
            console.info(row)
