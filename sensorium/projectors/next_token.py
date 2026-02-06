"""Next-token experiment figure projector.

Custom projector for the 3-panel next-token visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING

from sensorium.projectors.base import BaseProjector

if TYPE_CHECKING:
    from sensorium.observers.inference import InferenceObserver


@dataclass
class NextTokenFigureConfig:
    """Configuration for next-token figure.
    
    Attributes:
        name: Output filename (without extension)
        segment_size: Segment size for position calculations
        format: Output format (png, pdf, svg)
        dpi: Output DPI
    """
    name: str = "next_token"
    segment_size: int = 16
    format: str = "png"
    dpi: int = 300


class NextTokenFigureProjector(BaseProjector):
    """Custom projector for next-token experiment figures.
    
    Generates a 3-panel visualization:
    A) Trie branching (node-link diagram)
    B) Probability distributions at branch points
    C) Accuracy by position
    
    Example:
        projector = NextTokenFigureProjector(NextTokenFigureConfig(
            name="next_token",
            segment_size=16,
        ))
        projector.project(inference_observer)
    """
    
    def __init__(
        self,
        config: NextTokenFigureConfig | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Figure configuration
            output_dir: Output directory for figures
            **kwargs: Shortcut for config fields
        """
        super().__init__(output_dir or Path("paper/figures"))
        
        if config:
            self.config = config
        else:
            self.config = NextTokenFigureConfig(**kwargs)
    
    def project(self, source: Union["InferenceObserver", Dict[str, Any]]) -> Dict[str, Any]:
        """Generate the 3-panel figure.
        
        Args:
            source: InferenceObserver or dict with predictions data
        
        Returns:
            Dict with status and output path
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Extract data from source
        data = self._extract_data(source)
        
        # Get predictions - try multiple field names
        predictions = data.get("predictions", [])
        if not predictions and "results" in data:
            # If using InferenceObserver with accumulated results
            results = data["results"]
            if results and isinstance(results[0], dict):
                predictions = results[0].get("predictions", [])
        
        accuracy = data.get("accuracy", 0.0)
        segment_size = self.config.segment_size
        
        if not predictions:
            return {"status": "skipped", "reason": "no predictions"}
        
        self.ensure_output_dir()
        
        # Build data structures for visualization
        prefix_to_continuations = self._build_continuation_map(predictions, segment_size)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel A: Trie branching visualization
        self._plot_trie_branching(axes[0], prefix_to_continuations)
        
        # Panel B: Probability distributions at branch points
        self._plot_probability_distributions(axes[1], prefix_to_continuations)
        
        # Panel C: Accuracy by position
        self._plot_accuracy_by_position(axes[2], predictions, segment_size, accuracy)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.config.name}.{self.config.format}"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return {"status": "success", "path": str(output_path)}
    
    def _build_continuation_map(
        self,
        predictions: List[Dict],
        segment_size: int,
    ) -> Dict[int, Dict[str, float]]:
        """Build position -> continuations map."""
        prefix_to_continuations: Dict[int, Dict[str, float]] = {}
        
        for p in predictions:
            pos = p["position"]
            scores = p["scores"]
            continuations = [(b, float(scores[b])) for b in range(256) if scores[b] > 0.01]
            
            if continuations:
                seg_pos = pos % segment_size
                if seg_pos not in prefix_to_continuations:
                    prefix_to_continuations[seg_pos] = {}
                
                for byte_val, score in continuations:
                    char = chr(byte_val) if 32 <= byte_val < 127 else f'x{byte_val:02x}'
                    if char not in prefix_to_continuations[seg_pos]:
                        prefix_to_continuations[seg_pos][char] = 0
                    prefix_to_continuations[seg_pos][char] += score
        
        return prefix_to_continuations
    
    def _plot_trie_branching(self, ax, prefix_to_continuations: Dict):
        """Plot panel A: Trie branching visualization."""
        import matplotlib.pyplot as plt
        
        positions_to_show = sorted(prefix_to_continuations.keys())[:14]
        y_positions = {}
        
        for i, pos in enumerate(positions_to_show):
            x = i * 1.0
            continuations = prefix_to_continuations.get(pos, {})
            sorted_conts = sorted(continuations.items(), key=lambda x: -x[1])[:4]
            total_score = sum(s for _, s in sorted_conts)
            
            for j, (char, score) in enumerate(sorted_conts):
                y = 0.5 - j * 0.25
                prob = score / total_score if total_score > 0 else 0
                
                size = 150 + prob * 350
                color = plt.cm.Blues(0.3 + prob * 0.7)
                
                ax.scatter([x], [y], s=size, c=[color], edgecolors='black', linewidths=0.5, zorder=3)
                ax.annotate(char, (x, y), ha='center', va='center', fontsize=7, fontweight='bold')
                
                if i > 0 and pos - 1 in y_positions:
                    for prev_y in y_positions[pos - 1]:
                        ax.plot([x-1, x], [prev_y, y], 'k-', alpha=0.15, linewidth=0.5)
            
            y_positions[pos] = [0.5 - j * 0.25 for j in range(len(sorted_conts))]
        
        ax.set_xlim(-0.5, len(positions_to_show) - 0.5)
        ax.set_ylim(-0.6, 0.75)
        ax.set_xlabel('Position in segment', fontsize=10)
        ax.set_xticks(range(0, len(positions_to_show), 2))
        ax.set_xticklabels([positions_to_show[i] for i in range(0, len(positions_to_show), 2)], fontsize=8)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    def _plot_probability_distributions(self, ax, prefix_to_continuations: Dict):
        """Plot panel B: Probability distributions at branch points."""
        import matplotlib.pyplot as plt
        
        interesting_positions = []
        for pos, conts in prefix_to_continuations.items():
            sorted_conts = sorted(conts.items(), key=lambda x: -x[1])
            if len(sorted_conts) >= 2:
                total = sum(s for _, s in sorted_conts)
                probs = [s/total for _, s in sorted_conts[:4]]
                if probs[0] < 0.85:
                    interesting_positions.append((pos, sorted_conts[:4], probs))
        
        interesting_positions = sorted(
            interesting_positions,
            key=lambda x: -x[2][1] if len(x[2]) > 1 else 0
        )[:5]
        
        if interesting_positions:
            for i, (pos, conts, probs) in enumerate(interesting_positions):
                bottom = 0
                for j, ((char, _), prob) in enumerate(zip(conts, probs)):
                    color = plt.cm.Set2(j)
                    ax.bar(i, prob, 0.65, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
                    if prob > 0.12:
                        ax.text(i, bottom + prob/2, f"'{char}'\n{prob:.0%}", 
                               ha='center', va='center', fontsize=7, fontweight='bold')
                    bottom += prob
            
            ax.set_xticks(range(len(interesting_positions)))
            ax.set_xticklabels([f"pos {p[0]}" for p in interesting_positions], fontsize=9)
        
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    def _plot_accuracy_by_position(self, ax, predictions: List[Dict], segment_size: int, accuracy: float):
        """Plot panel C: Accuracy by position."""
        pos_correct: Dict[int, int] = {}
        pos_total: Dict[int, int] = {}
        
        for p in predictions:
            seg_pos = p["position"] % segment_size
            if seg_pos not in pos_correct:
                pos_correct[seg_pos] = 0
                pos_total[seg_pos] = 0
            pos_total[seg_pos] += 1
            if p["predicted"] == p["actual"]:
                pos_correct[seg_pos] += 1
        
        positions = sorted(pos_correct.keys())
        accuracies = [pos_correct[p] / pos_total[p] if pos_total[p] > 0 else 0 for p in positions]
        
        colors = ['#e74c3c' if a < 0.7 else '#f39c12' if a < 0.95 else '#27ae60' for a in accuracies]
        ax.bar(positions, accuracies, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=accuracy, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {accuracy:.1%}')
        ax.set_xlabel('Position in segment', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
