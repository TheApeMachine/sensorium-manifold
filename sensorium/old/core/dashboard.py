"""Real-time Dashboard for Thermodynamic Manifold Visualization

Provides live visualization of manifold internals during training:
- Bond graph structure and dynamics
- Energy and heat distributions
- Excitation flow patterns
- Prediction confidence

Usage:
    from thermo_manifold.core.dashboard import Dashboard
    
    dashboard = Dashboard(manifold, vocab=vocab)
    dashboard.start()
    
    # During training loop:
    for step in range(steps):
        train_step(...)
        dashboard.update(step=step, extra_info={"accuracy": acc})
    
    dashboard.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..semantic.manifold import SemanticManifold


@dataclass
class DashboardState:
    """Snapshot of manifold state for visualization."""
    step: int
    
    # Graph stats
    num_edges: int
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weights: torch.Tensor
    
    # Attractor stats
    excitation: torch.Tensor
    heat: torch.Tensor
    energy: torch.Tensor
    
    # Prediction stats
    entropy: float
    top_k_tokens: List[str]
    top_k_probs: List[float]
    
    # Extra info from caller
    extra: Dict[str, Any]


class Dashboard:
    """Real-time visualization dashboard for the thermodynamic manifold."""
    
    def __init__(
        self,
        manifold: "SemanticManifold",
        vocab: Optional[List[str]] = None,
        update_interval: float = 0.5,  # seconds between redraws
        figsize: tuple = (16, 10),
    ):
        self.manifold = manifold
        self.vocab = vocab or [f"tok_{i}" for i in range(manifold.vocab_size)]
        self.update_interval = update_interval
        self.figsize = figsize
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state: Optional[DashboardState] = None
        self._lock = threading.Lock()
        
        # History for time series
        self._history: Dict[str, List[float]] = {
            "step": [],
            "num_edges": [],
            "total_energy": [],
            "total_heat": [],
            "entropy": [],
        }
        self._max_history = 500  # Keep last N points
        
        # Matplotlib objects (lazy init)
        self._fig = None
        self._axes = None
        self._warned_non_interactive = False
    
    def _init_figure(self):
        """Initialize matplotlib figure and axes."""
        import matplotlib
        self._ensure_interactive_backend(matplotlib)
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.ion()  # Interactive mode
        
        self._fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        self._axes = {
            "graph": axes[0, 0],       # Bond graph visualization
            "energy": axes[0, 1],      # Energy/heat bar chart
            "top_tokens": axes[0, 2],  # Top active tokens
            "timeseries": axes[1, 0],  # Metrics over time
            "weights": axes[1, 1],     # Weight distribution
            "predictions": axes[1, 2], # Current predictions
        }
        
        self._fig.suptitle("Thermodynamic Manifold Dashboard", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    def _ensure_interactive_backend(self, matplotlib_module) -> None:
        """Try to switch to an interactive backend if possible."""
        interactive_backends = {
            "MacOSX",
            "TkAgg",
            "QtAgg",
            "Qt5Agg",
            "GTK3Agg",
            "WXAgg",
        }
        current = matplotlib_module.get_backend()
        if current in interactive_backends:
            return

        for candidate in ("MacOSX", "TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib_module.use(candidate, force=True)
                return
            except Exception:
                continue

        if not self._warned_non_interactive:
            print(
                f"Dashboard: non-interactive matplotlib backend '{current}'. "
                "Real-time window may not appear."
            )
            self._warned_non_interactive = True
    
    def _capture_state(self, step: int, extra: Dict[str, Any]) -> DashboardState:
        """Capture current manifold state."""
        graph = self.manifold.graph
        attractors = self.manifold.attractors
        
        # Get graph edges
        if graph.num_edges > 0:
            edge_src = graph.src.detach().cpu()
            edge_dst = graph.dst.detach().cpu()
            edge_weights = graph.w.detach().cpu()
        else:
            edge_src = torch.tensor([], dtype=torch.long)
            edge_dst = torch.tensor([], dtype=torch.long)
            edge_weights = torch.tensor([], dtype=torch.float32)
        
        # Get attractor states
        excitation = attractors.get("excitation").detach().cpu()
        heat = attractors.get("heat").detach().cpu()
        energy = attractors.get("energy").detach().cpu()
        
        # Get prediction state
        out = self.manifold.output_state()
        entropy = out.meta.get("entropy", 0.0)
        
        # Top-k predictions
        k = min(10, len(self.vocab))
        top_probs, top_idx = torch.topk(out.probs.cpu(), k)
        top_tokens = [self.vocab[i] if i < len(self.vocab) else f"?{i}" for i in top_idx.tolist()]
        top_k_probs = top_probs.tolist()
        
        return DashboardState(
            step=step,
            num_edges=graph.num_edges,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weights=edge_weights,
            excitation=excitation,
            heat=heat,
            energy=energy,
            entropy=entropy,
            top_k_tokens=top_tokens,
            top_k_probs=top_k_probs,
            extra=extra,
        )
    
    def update(self, step: int = 0, extra: Optional[Dict[str, Any]] = None) -> None:
        """Update dashboard with current manifold state.
        
        Call this from your training loop.
        """
        extra = extra or {}
        
        with self._lock:
            self._state = self._capture_state(step, extra)
            
            # Update history
            self._history["step"].append(step)
            self._history["num_edges"].append(self._state.num_edges)
            self._history["total_energy"].append(float(self._state.energy.sum().item()))
            self._history["total_heat"].append(float(self._state.heat.sum().item()))
            self._history["entropy"].append(self._state.entropy)
            
            # Add extra metrics to history
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    if k not in self._history:
                        self._history[k] = []
                    self._history[k].append(v)
            
            # Trim history
            for k in self._history:
                if len(self._history[k]) > self._max_history:
                    self._history[k] = self._history[k][-self._max_history:]
    
    def _render(self) -> None:
        """Render the dashboard (called from background thread)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        with self._lock:
            state = self._state
            history = {k: list(v) for k, v in self._history.items()}  # Copy
        
        if state is None:
            return
        
        # Clear all axes
        for ax in self._axes.values():
            ax.clear()
        
        # 1. Bond graph (simplified - show edge count and top connections)
        ax = self._axes["graph"]
        if state.num_edges > 0:
            # Show top edges by weight
            top_k = min(20, len(state.edge_weights))
            if top_k > 0:
                top_idx = torch.topk(state.edge_weights, top_k).indices
                for i in top_idx[:10]:
                    src = state.edge_src[i].item()
                    dst = state.edge_dst[i].item()
                    w = state.edge_weights[i].item()
                    src_name = self.vocab[src][:8] if src < len(self.vocab) else f"?{src}"
                    dst_name = self.vocab[dst][:8] if dst < len(self.vocab) else f"?{dst}"
                    ax.text(0.1, 0.9 - i * 0.08, f"{src_name} → {dst_name}: {w:.3f}", 
                            fontsize=8, transform=ax.transAxes, family='monospace')
        ax.set_title(f"Top Bonds ({state.num_edges} total edges)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 2. Energy/Heat distribution (top active tokens)
        ax = self._axes["energy"]
        top_k = 15
        exc = state.excitation
        heat = state.heat
        
        # Find top excited tokens
        top_exc_idx = torch.topk(exc, min(top_k, len(exc))).indices
        labels = [self.vocab[i][:6] if i < len(self.vocab) else f"?{i}" for i in top_exc_idx.tolist()]
        exc_vals = exc[top_exc_idx].numpy()
        heat_vals = heat[top_exc_idx].numpy()
        
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, exc_vals, width, label='Excitation', color='steelblue')
        ax.bar(x + width/2, heat_vals, width, label='Heat', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title("Top Active Tokens")
        ax.legend(fontsize=8)
        
        # 3. Top predictions
        ax = self._axes["top_tokens"]
        if state.top_k_tokens:
            y = np.arange(len(state.top_k_tokens))
            ax.barh(y, state.top_k_probs, color='green', alpha=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels([t[:10] for t in state.top_k_tokens], fontsize=8)
            ax.set_xlabel("Probability")
            ax.set_title(f"Top Predictions (H={state.entropy:.2f})")
            ax.set_xlim(0, max(0.5, max(state.top_k_probs) * 1.1))
        
        # 4. Time series
        ax = self._axes["timeseries"]
        if history["step"]:
            steps = history["step"]
            ax.plot(steps, history["num_edges"], label="Edges", alpha=0.8)
            
            # Normalize for same scale
            if max(history["total_energy"]) > 0:
                energy_norm = [e / max(history["total_energy"]) * max(history["num_edges"]) 
                               for e in history["total_energy"]]
                ax.plot(steps, energy_norm, label="Energy (scaled)", alpha=0.6)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Count / Value")
            ax.set_title("Dynamics Over Time")
            ax.legend(fontsize=8, loc='upper left')
        
        # 5. Weight distribution
        ax = self._axes["weights"]
        if state.num_edges > 0:
            weights = state.edge_weights.numpy()
            ax.hist(weights, bins=30, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Bond Weight")
            ax.set_ylabel("Count")
            ax.set_title(f"Weight Distribution (μ={weights.mean():.3f})")
            ax.axvline(weights.mean(), color='red', linestyle='--', alpha=0.7)
        
        # 6. Current state summary
        ax = self._axes["predictions"]
        info_lines = [
            f"Step: {state.step}",
            f"Edges: {state.num_edges}",
            f"Total Excitation: {state.excitation.sum():.2f}",
            f"Total Heat: {state.heat.sum():.2f}",
            f"Entropy: {state.entropy:.3f}",
            "",
        ]
        # Add extra info
        for k, v in state.extra.items():
            if isinstance(v, float):
                info_lines.append(f"{k}: {v:.4f}")
            else:
                info_lines.append(f"{k}: {v}")
        
        ax.text(0.1, 0.9, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        ax.set_title("Current State")
        ax.axis('off')
        
        # Redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
    
    def _render_loop(self) -> None:
        """Background thread that periodically renders."""
        import matplotlib.pyplot as plt
        
        self._init_figure()
        
        while self._running:
            try:
                self._render()
            except Exception as e:
                print(f"Dashboard render error: {e}")
            
            time.sleep(self.update_interval)
        
        plt.ioff()
    
    def start(self) -> None:
        """Start the dashboard in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        
        # Give matplotlib time to initialize
        time.sleep(0.5)
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def save_snapshot(self, path: str) -> None:
        """Save current dashboard state as an image."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Dashboard snapshot saved to {path}")


class SimpleDashboard:
    """Advanced dashboard for thermodynamic manifold visualization.
    
    Shows dynamic interactions:
    - 3D bond graph with particles, bonds, and heat cloud
    - Label token competition dynamics over time
    - Token trajectory trails in energy/heat/excitation space
    - Flow visualization showing energy movement
    - Heat diffusion patterns
    - Energy transfer between regions
    """
    
    def __init__(
        self,
        manifold: "SemanticManifold",
        vocab: Optional[List[str]] = None,
        figsize: tuple = (20, 12),
        show_3d: bool = True,
    ):
        self.manifold = manifold
        self.vocab = vocab or [f"tok_{i}" for i in range(manifold.vocab_size)]
        self.figsize = figsize
        self.show_3d = show_3d
        
        self._fig = None
        self._axes = None
        self._ax3d = None
        self._warned_non_interactive = False
        self._heat_sample_idx = None
        self._positions_3d = None
        self._pca_components = None  # For projecting carriers to same space
        self._pca_mean = None
        
        # Find label token indices
        self._label_indices = [i for i, v in enumerate(self.vocab) if v.startswith("label_")]
        
        # History for time series and trajectories
        self._history: Dict[str, List[float]] = {
            "step": [],
            "num_edges": [],
            "entropy": [],
            "total_energy": [],
            "total_heat": [],
            "temperature": [],
            "efficiency": [],
            "loss": [],       # Cross-entropy loss
            "ppl": [],        # Perplexity
        }
        
        # Label competition history (energy, heat, excitation for each label)
        self._label_history: Dict[str, List[List[float]]] = {
            "energy": [],      # List of [label0, label1, ..., label9] per step
            "heat": [],
            "excitation": [],
            "probs": [],
        }
        
        # Trajectory tracking for selected tokens
        self._trajectory_tokens: List[int] = []  # Will be set to label indices
        self._trajectories: Dict[int, Dict[str, List[float]]] = {}  # token_id -> {energy: [...], heat: [...], exc: [...]}
        
        # Flow history for animation
        self._flow_history: List[torch.Tensor] = []
        self._max_flow_history = 10
        
        # New edge tracking
        self._prev_num_edges = 0
        self._new_edges_history: List[int] = []
        
        # Chunk tracking (if manifold has chunks)
        self._chunk_history: Dict[str, List[float]] = {
            "num_chunks": [],
            "chunk_energy": [],
            "chunk_excitation": [],
            "chunk_bonds": [],
        }
        self._chunk_positions_3d = None  # Will be computed if chunks exist
    
    def _compute_3d_positions(self) -> None:
        """Compute 3D positions for all nodes via PCA.
        
        Also stores the PCA transformation for projecting carriers.
        """
        import numpy as np
        
        positions = self.manifold.attractors.get("position").detach().float().cpu().numpy()
        
        if positions.shape[1] > 3:
            self._pca_mean = positions.mean(axis=0)
            positions_centered = positions - self._pca_mean
            try:
                u, s, vh = np.linalg.svd(positions_centered, full_matrices=False)
                self._positions_3d = u[:, :3] * s[:3]
                # Store the projection matrix for carriers
                self._pca_components = vh[:3, :]  # [3, D]
                self._pca_scale = s[:3]
            except Exception:
                self._positions_3d = positions[:, :3]
                self._pca_components = None
                self._pca_mean = None
        else:
            self._positions_3d = positions[:, :3] if positions.shape[1] >= 3 else np.pad(
                positions, ((0, 0), (0, 3 - positions.shape[1]))
            )
            self._pca_components = None
            self._pca_mean = None
    
    def _project_to_3d(self, positions: 'np.ndarray') -> 'np.ndarray':
        """Project positions to 3D using the same PCA as particles."""
        import numpy as np
        
        if self._pca_components is not None and self._pca_mean is not None:
            # Use same PCA transformation as particles
            centered = positions - self._pca_mean
            return centered @ self._pca_components.T
        elif positions.shape[1] >= 3:
            return positions[:, :3]
        else:
            return np.pad(positions, ((0, 0), (0, 3 - positions.shape[1])))
    
    def update_and_render(self, step: int = 0, extra: Optional[Dict[str, Any]] = None) -> None:
        """Update state and render with consolidated 2-panel layout."""
        import matplotlib
        self._ensure_interactive_backend(matplotlib)
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyBboxPatch, Rectangle
        from matplotlib.colors import LinearSegmentedColormap
        
        extra = extra or {}
        
        if self._fig is None:
            plt.ion()
            # Two-panel layout: 3D visualization (left), Combined metrics (right)
            self._fig = plt.figure(figsize=(18, 10))
            
            # Left: Large 3D manifold visualization (takes ~60% width)
            self._ax3d = self._fig.add_subplot(1, 2, 1, projection='3d')
            
            # Right: Combined metrics panel
            self._axes = {
                "metrics": self._fig.add_subplot(1, 2, 2),
            }
            plt.show(block=False)
            self._compute_3d_positions()
            
            # Initialize trajectory tracking for label tokens
            self._trajectory_tokens = self._label_indices[:5]
            for tok in self._trajectory_tokens:
                self._trajectories[tok] = {"energy": [], "heat": [], "exc": []}
        
        # Capture state
        graph = self.manifold.graph
        attractors = self.manifold.attractors
        out = self.manifold.output_state()
        energy = attractors.get("energy").detach().float().cpu()
        heat = attractors.get("heat").detach().float().cpu()
        excitation = attractors.get("excitation").detach().float().cpu()
        probs = out.probs.detach().float().cpu()
        
        total_energy = float(energy.sum().item())
        total_heat = float(heat.sum().item())
        energy_input = getattr(self.manifold, '_total_energy_input', 0.0)
        conservation_error = abs(total_energy + total_heat - energy_input) / (energy_input + 1e-8)
        
        # Track new edges
        new_edges = graph.num_edges - self._prev_num_edges
        self._prev_num_edges = graph.num_edges
        self._new_edges_history.append(new_edges)
        if len(self._new_edges_history) > 100:
            self._new_edges_history = self._new_edges_history[-100:]
        
        # Update label history
        if self._label_indices:
            self._label_history["energy"].append(energy[self._label_indices].numpy().tolist())
            self._label_history["heat"].append(heat[self._label_indices].numpy().tolist())
            self._label_history["excitation"].append(excitation[self._label_indices].numpy().tolist())
            self._label_history["probs"].append(probs[self._label_indices].numpy().tolist())
            for key in self._label_history:
                if len(self._label_history[key]) > 200:
                    self._label_history[key] = self._label_history[key][-200:]
        
        # Update trajectory history
        for tok in self._trajectory_tokens:
            self._trajectories[tok]["energy"].append(float(energy[tok].item()))
            self._trajectories[tok]["heat"].append(float(heat[tok].item()))
            self._trajectories[tok]["exc"].append(float(excitation[tok].item()))
            for key in self._trajectories[tok]:
                if len(self._trajectories[tok][key]) > 100:
                    self._trajectories[tok][key] = self._trajectories[tok][key][-100:]
        
        # Update basic history
        if "energy_input" not in self._history:
            self._history["energy_input"] = []
        self._history["step"].append(step)
        self._history["num_edges"].append(graph.num_edges)
        self._history["entropy"].append(out.meta.get("entropy", 0.0))
        self._history["total_energy"].append(total_energy)
        self._history["total_heat"].append(total_heat)
        self._history["energy_input"].append(energy_input)
        
        # Track loss and perplexity from extra metrics
        if extra is not None:
            self._history["loss"].append(extra.get("loss", 0.0))
            self._history["ppl"].append(extra.get("ppl", 1.0))
        else:
            # Fallback: use entropy as proxy for loss
            self._history["loss"].append(out.meta.get("entropy", 0.0))
            self._history["ppl"].append(2.0 ** out.meta.get("entropy", 0.0))
        
        # Update chunk history (if manifold has chunks)
        chunks = getattr(self.manifold, 'chunks', None)
        chunk_graph = getattr(self.manifold, 'chunk_graph', None)
        if chunks is not None:
            num_chunks = chunks.num_chunks
            chunk_energy = float(chunks.energy.sum().item()) if num_chunks > 0 else 0.0
            chunk_exc = float(chunks.excitation.sum().item()) if num_chunks > 0 else 0.0
            chunk_bonds = chunk_graph.num_edges if chunk_graph is not None else 0
            
            self._chunk_history["num_chunks"].append(num_chunks)
            self._chunk_history["chunk_energy"].append(chunk_energy)
            self._chunk_history["chunk_excitation"].append(chunk_exc)
            self._chunk_history["chunk_bonds"].append(chunk_bonds)
            
            for key in self._chunk_history:
                if len(self._chunk_history[key]) > 200:
                    self._chunk_history[key] = self._chunk_history[key][-200:]
        
        # Clear axes
        for ax in self._axes.values():
            ax.clear()
        if self._ax3d:
            self._ax3d.clear()
        
        # ===== LEFT PANEL: 3D Manifold with nodes, bonds, heat mesh, flow =====
        if graph.num_edges > 0:
            self._render_3d_unified(energy, heat, excitation, graph, step)
        
        # ===== RIGHT PANEL: Combined Metrics Dashboard =====
        ax = self._axes["metrics"]
        self._render_metrics_panel(ax, step, energy, heat, excitation, probs, 
                                   total_energy, total_heat, energy_input, 
                                   conservation_error, new_edges, extra)
        
        # Title
        self._fig.suptitle(f"Thermodynamic Manifold — Step {step}", fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _render_3d_unified(self, energy: torch.Tensor, heat: torch.Tensor,
                           excitation: torch.Tensor, graph, step: int) -> None:
        """Render unified 3D view with full topology visualization.
        
        Shows:
        1. Particles colored by energy, sized by excitation
        2. Carriers colored by energy, with heat glow
        3. Particle-carrier bonds with energy flow coloring
        4. Bond thickness = strength, color = energy flowing through
        5. Directional arrows showing energy flow
        6. Heat radiation as glow around hot elements
        7. Label tokens as stars
        8. Chunks as diamonds
        """
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if self._positions_3d is None:
            self._compute_3d_positions()
        
        ax = self._ax3d
        positions_3d = self._positions_3d
        energy_np = energy.numpy()
        heat_np = heat.numpy()
        exc_np = excitation.numpy()
        
        max_nodes = 150
        
        # Find active particles (those with energy or excitation)
        activity = energy_np + exc_np + heat_np
        active_nodes = np.where(activity > np.percentile(activity, 70))[0]
        if len(active_nodes) > max_nodes:
            top_idx = np.argsort(activity[active_nodes])[::-1][:max_nodes]
            active_nodes = active_nodes[top_idx]
        
        active_set = set(active_nodes)
        
        # Get carrier and bond info
        carriers = getattr(self.manifold, 'carriers', None)
        pc_bonds = getattr(self.manifold, 'particle_carrier_bonds', None)
        carrier_3d = None
        num_carriers = 0
        carrier_energy_np = None
        carrier_heat_np = None
        
        # ===== 1. Draw heat radiation from particles (background glow) =====
        heat_threshold = np.percentile(heat_np, 85)
        hot_nodes = np.where(heat_np > heat_threshold)[0]
        hot_nodes = [n for n in hot_nodes if n in active_set]
        
        if len(hot_nodes) > 0:
            hot_positions = positions_3d[hot_nodes]
            hot_heat = heat_np[hot_nodes]
            hot_sizes = 200 + 800 * (hot_heat / (hot_heat.max() + 1e-8))
            
            # Multiple layers for glow effect (outermost first)
            for alpha, size_mult, color in [
                (0.05, 3.0, 'red'),
                (0.08, 2.2, 'orangered'),
                (0.12, 1.5, 'orange'),
                (0.18, 1.0, 'yellow')
            ]:
                ax.scatter(
                    hot_positions[:, 0], hot_positions[:, 1], hot_positions[:, 2],
                    c=color, s=hot_sizes * size_mult, alpha=alpha,
                    edgecolors='none', marker='o'
                )
        
        # ===== 2. Draw carriers with energy coloring and heat glow =====
        if carriers is not None and carriers.num_carriers > 0:
            num_carriers = carriers.num_carriers
            carrier_pos = carriers.position.detach().cpu().numpy()
            carrier_energy_np = carriers.energy.detach().cpu().numpy()
            carrier_heat_np = carriers.heat.detach().cpu().numpy()
            
            # Project carrier positions to 3D using SAME transformation as particles
            carrier_3d = self._project_to_3d(carrier_pos)
            
            # Heat glow around carriers
            carrier_heat_max = carrier_heat_np.max() + 1e-8
            hot_carriers = np.where(carrier_heat_np > carrier_heat_max * 0.3)[0]
            if len(hot_carriers) > 0:
                for alpha, size_mult in [(0.06, 2.5), (0.12, 1.5)]:
                    ax.scatter(
                        carrier_3d[hot_carriers, 0], 
                        carrier_3d[hot_carriers, 1], 
                        carrier_3d[hot_carriers, 2],
                        c='orange', 
                        s=300 * size_mult * (carrier_heat_np[hot_carriers] / carrier_heat_max),
                        alpha=alpha, edgecolors='none', marker='o'
                    )
            
            # Draw carriers - size by energy, color by heat
            carrier_energy_max = carrier_energy_np.max() + 1e-8
            carrier_sizes = 80 + 350 * (carrier_energy_np / carrier_energy_max)
            
            # Color by energy (blue=low, red=high)
            carrier_colors = carrier_energy_np / carrier_energy_max
            
            ax.scatter(
                carrier_3d[:, 0], carrier_3d[:, 1], carrier_3d[:, 2],
                c=carrier_colors, cmap='coolwarm', s=carrier_sizes,
                marker='h', alpha=0.85, edgecolors='white', linewidths=1.0,
                vmin=0, vmax=1, label='Carriers', zorder=50
            )
            
            # ===== 3. Draw particle-carrier bonds with energy flow =====
            if pc_bonds is not None and pc_bonds.num_bonds > 0:
                p_ids = pc_bonds.particle_ids.detach().cpu().numpy()
                c_ids = pc_bonds.carrier_ids.detach().cpu().numpy()
                strengths = pc_bonds.strengths.detach().cpu().numpy()
                
                # Get last energy flow if available
                if hasattr(pc_bonds, 'last_energy_flow') and pc_bonds.last_energy_flow.numel() > 0:
                    energy_flow = pc_bonds.last_energy_flow.detach().cpu().numpy()
                else:
                    energy_flow = strengths  # Fallback to strength
                
                # Sample bonds - show top by strength (not flow, since flow might be 0)
                max_bonds = 500
                if len(strengths) > max_bonds:
                    top_idx = np.argsort(strengths)[::-1][:max_bonds]
                    p_ids = p_ids[top_idx]
                    c_ids = c_ids[top_idx]
                    strengths = strengths[top_idx]
                    energy_flow = energy_flow[top_idx]
                
                strength_max = strengths.max() + 1e-8
                flow_max = max(energy_flow.max(), 1e-8)  # Avoid division by zero
                
                bonds_drawn = 0
                
                # Draw ALL top bonds (don't filter by active_set too aggressively)
                step_size = max(1, len(p_ids) // 200)
                for i in range(0, len(p_ids), step_size):
                    p, c = p_ids[i], c_ids[i]
                    s, f = strengths[i], energy_flow[i]
                    
                    if p >= len(positions_3d) or c >= len(carrier_3d):
                        continue
                    
                    p_pos = positions_3d[p]
                    c_pos = carrier_3d[c]
                    
                    # Line properties based on strength (since flow might be 0)
                    strength_ratio = s / strength_max
                    flow_ratio = f / flow_max if flow_max > 1e-8 else strength_ratio
                    
                    # Color: cyan (low) -> yellow -> orange (high) based on strength
                    if strength_ratio < 0.33:
                        r, g, b = 0.2, 0.8, 1.0  # Cyan
                    elif strength_ratio < 0.66:
                        r, g, b = 1.0, 1.0, 0.2  # Yellow
                    else:
                        r, g, b = 1.0, 0.5, 0.0  # Orange
                    
                    # More visible alpha
                    alpha = 0.3 + 0.5 * strength_ratio
                    linewidth = 0.5 + 3.0 * strength_ratio
                    
                    ax.plot(
                        [p_pos[0], c_pos[0]], 
                        [p_pos[1], c_pos[1]], 
                        [p_pos[2], c_pos[2]],
                        color=(r, g, b, alpha), linewidth=linewidth
                    )
                    bonds_drawn += 1
                    
                    # ===== 5. Draw directional arrows for strong bonds =====
                    if strength_ratio > 0.5:
                        direction = c_pos - p_pos
                        length = np.linalg.norm(direction)
                        if length > 0.01:
                            mid = p_pos + 0.5 * direction
                            arrow_len = 0.1 * length
                            ax.quiver(
                                mid[0], mid[1], mid[2],
                                direction[0] * arrow_len / length, 
                                direction[1] * arrow_len / length, 
                                direction[2] * arrow_len / length,
                                color=(r, g, b), alpha=0.8,
                                arrow_length_ratio=0.5, linewidth=1.5
                            )
                
                # Debug: print bond count
                # print(f"[DEBUG] Drew {bonds_drawn} bonds out of {pc_bonds.num_bonds} total")
        
        # ===== 4. Draw particles colored by energy, sized by excitation =====
        node_positions = positions_3d[active_nodes]
        node_energy = energy_np[active_nodes]
        node_heat = heat_np[active_nodes]
        node_exc = exc_np[active_nodes]
        
        # Size by excitation
        exc_max = node_exc.max() + 1e-8
        sizes = 25 + 180 * (node_exc / exc_max)
        
        # Color by energy (normalized)
        energy_max = energy_np.max() + 1e-8
        energy_colors = node_energy / energy_max
        
        ax.scatter(
            node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
            c=energy_colors, cmap='viridis', s=sizes, alpha=0.9,
            edgecolors='white', linewidths=0.3, vmin=0, vmax=1, zorder=60
        )
        
        # ===== 6. Draw heat radiation from particles (inner glow for high-heat particles) =====
        # Already done in step 1, but add particle-specific heat indicators
        for i, node_idx in enumerate(active_nodes):
            if heat_np[node_idx] > np.percentile(heat_np, 95):
                # Extra glow for very hot particles
                ax.scatter(
                    [node_positions[i, 0]], [node_positions[i, 1]], [node_positions[i, 2]],
                    c='white', s=sizes[i] * 0.3, alpha=0.9, edgecolors='none', zorder=61
                )
        
        # ===== Legacy token-token edges (if present) =====
        edge_src = np.array([])
        edge_weights = np.array([])
        if graph.num_edges > 0:
            edge_src = graph.src.detach().cpu().numpy()
            edge_dst = graph.dst.detach().cpu().numpy()
            edge_weights = graph.w.detach().cpu().numpy()
            
            max_edges = 80
            if len(edge_weights) > max_edges:
                edge_order = np.argsort(edge_weights)[::-1][:max_edges]
                edge_src = edge_src[edge_order]
                edge_dst = edge_dst[edge_order]
                edge_weights = edge_weights[edge_order]
            
            edge_weight_max = edge_weights.max() + 1e-8
            
            for i in range(0, len(edge_src), max(1, len(edge_src) // 40)):
                src, dst, w = edge_src[i], edge_dst[i], edge_weights[i]
                if src in active_set and dst in active_set:
                    x = [positions_3d[src, 0], positions_3d[dst, 0]]
                    y = [positions_3d[src, 1], positions_3d[dst, 1]]
                    z = [positions_3d[src, 2], positions_3d[dst, 2]]
                    
                    strength = w / edge_weight_max
                    color = (0.6, 0.6, 0.6, 0.15 + 0.25*strength)
                    ax.plot(x, y, z, color=color, linewidth=0.4 + strength)
        
        # ===== 7. Label tokens as bright stars =====
        label_in_active = [n for n in active_nodes if n in self._label_indices]
        if label_in_active:
            label_positions = positions_3d[label_in_active]
            label_energy = energy_np[label_in_active]
            label_heat = heat_np[label_in_active]
            label_sizes = 400 + 600 * (label_energy / (label_energy.max() + 1e-8))
            
            # Heat glow for labels
            for i, node_idx in enumerate(label_in_active):
                if heat_np[node_idx] > 0:
                    glow_size = 150 + 400 * (label_heat[i] / (heat_np.max() + 1e-8))
                    ax.scatter(
                        [label_positions[i, 0]], [label_positions[i, 1]], [label_positions[i, 2]],
                        c='yellow', s=glow_size, alpha=0.3, edgecolors='none', zorder=99
                    )
            
            ax.scatter(
                label_positions[:, 0], label_positions[:, 1], label_positions[:, 2],
                c='lime', s=label_sizes, marker='*', 
                edgecolors='darkgreen', linewidths=2, alpha=0.95, zorder=100
            )
            
            # Label numbers with energy info
            for i, node_idx in enumerate(label_in_active):
                digit = self._label_indices.index(node_idx)
                e_val = energy_np[node_idx]
                ax.text(
                    label_positions[i, 0] + 0.02, label_positions[i, 1], 
                    label_positions[i, 2] + 0.02,
                    f"{digit}", fontsize=10, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='darkgreen', alpha=0.7)
                )
        
        # ===== Draw chunks as larger spheres (if available) =====
        chunks = getattr(self.manifold, 'chunks', None)
        chunk_graph = getattr(self.manifold, 'chunk_graph', None)
        num_chunks = 0
        num_chunk_bonds = 0
        
        if chunks is not None and chunks.num_chunks > 0:
            num_chunks = chunks.num_chunks
            chunk_pos = chunks.position.detach().cpu().numpy()
            chunk_energy = chunks.energy.detach().cpu().numpy()
            chunk_exc = chunks.excitation.detach().cpu().numpy()
            
            # Project chunk positions to 3D using same PCA basis
            if chunk_pos.shape[1] > 3:
                chunk_pos_centered = chunk_pos - chunk_pos.mean(axis=0)
                try:
                    u, s, vh = np.linalg.svd(chunk_pos_centered, full_matrices=False)
                    chunk_3d = u[:, :3] * s[:3]
                except Exception:
                    chunk_3d = chunk_pos[:, :3]
            else:
                chunk_3d = chunk_pos[:, :3] if chunk_pos.shape[1] >= 3 else np.pad(
                    chunk_pos, ((0, 0), (0, 3 - chunk_pos.shape[1]))
                )
            
            # Sample top chunks by excitation
            max_chunks = min(30, num_chunks)
            if num_chunks > max_chunks:
                top_idx = np.argsort(chunk_exc)[::-1][:max_chunks]
                chunk_3d = chunk_3d[top_idx]
                chunk_energy = chunk_energy[top_idx]
                chunk_exc = chunk_exc[top_idx]
            
            # Draw chunks as larger cubes/diamonds
            chunk_sizes = 200 + 600 * (chunk_exc / (chunk_exc.max() + 1e-8))
            ax.scatter(
                chunk_3d[:, 0], chunk_3d[:, 1], chunk_3d[:, 2],
                c='gold', s=chunk_sizes, marker='D',
                edgecolors='darkorange', linewidths=2, alpha=0.85, zorder=90
            )
            
            # Draw chunk->token bonds if available
            if chunk_graph is not None and chunk_graph.num_edges > 0:
                num_chunk_bonds = chunk_graph.num_edges
                chunk_src = chunk_graph.src.detach().cpu().numpy()
                chunk_dst = chunk_graph.dst.detach().cpu().numpy()
                chunk_w = chunk_graph.w.detach().cpu().numpy()
                
                # Sample top bonds
                max_c_edges = min(50, len(chunk_w))
                if len(chunk_w) > max_c_edges:
                    edge_order = np.argsort(chunk_w)[::-1][:max_c_edges]
                    chunk_src = chunk_src[edge_order]
                    chunk_dst = chunk_dst[edge_order]
                    chunk_w = chunk_w[edge_order]
                
                # Draw chunk->token edges as gold dashed lines
                c_weight_max = chunk_w.max() + 1e-8
                for i in range(0, len(chunk_src), max(1, len(chunk_src) // 30)):
                    s, d, w = chunk_src[i], chunk_dst[i], chunk_w[i]
                    if s < len(chunk_3d) and d in active_set:
                        x = [chunk_3d[s, 0], positions_3d[d, 0]]
                        y = [chunk_3d[s, 1], positions_3d[d, 1]]
                        z = [chunk_3d[s, 2], positions_3d[d, 2]]
                        alpha = 0.2 + 0.5 * (w / c_weight_max)
                        ax.plot(x, y, z, color='gold', linestyle='--', alpha=alpha, linewidth=1)
        
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)
        
        # Build title with stats
        num_pc_bonds = pc_bonds.num_bonds if pc_bonds is not None else 0
        
        # Energy stats
        total_particle_energy = energy_np.sum()
        total_carrier_energy = carrier_energy_np.sum() if carrier_energy_np is not None else 0
        total_particle_heat = heat_np.sum()
        total_carrier_heat = carrier_heat_np.sum() if carrier_heat_np is not None else 0
        
        line1 = f'{len(active_nodes)} particles | ⬡{num_carriers} carriers | {num_pc_bonds} bonds | ◆{num_chunks} chunks'
        line2 = f'E: {total_particle_energy:.1f}+{total_carrier_energy:.1f} | H: {total_particle_heat:.1f}+{total_carrier_heat:.1f}'
        line3 = '●=particles (color=energy) | ⬡=carriers | ★=labels | ◆=chunks | lines=bonds (color=flow)'
        
        ax.set_title(f'{line1}\n{line2}\n{line3}', fontsize=8)
        ax.set_facecolor('#1a1a2e')  # Dark background
    
    def _render_metrics_panel(self, ax, step: int, energy: torch.Tensor, heat: torch.Tensor,
                               excitation: torch.Tensor, probs: torch.Tensor,
                               total_energy: float, total_heat: float, energy_input: float,
                               conservation_error: float, new_edges: int,
                               extra: Optional[Dict[str, Any]] = None) -> None:
        """Render combined metrics panel with bullet charts and layered visualizations."""
        import numpy as np
        from matplotlib.patches import Rectangle, FancyBboxPatch
        from matplotlib.collections import PatchCollection
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Background
        ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#f8f9fa', edgecolor='none'))
        
        y_cursor = 9.5
        
        # ===== Section 1: Conservation & Thermodynamics (Bullet Chart) =====
        ax.text(0.2, y_cursor, "THERMODYNAMICS", fontsize=11, fontweight='bold', color='#2c3e50')
        y_cursor -= 0.4
        
        # Conservation bullet
        total_current = total_energy + total_heat
        conservation_pct = min(1.0, total_current / (energy_input + 1e-8))
        self._draw_bullet_chart(ax, 0.3, y_cursor - 0.3, 4.5, 0.35,
                               value=conservation_pct, 
                               target=1.0,
                               ranges=[0.9, 0.95, 1.0],
                               label="Conservation",
                               value_text=f"{conservation_pct:.1%}")
        
        # Energy vs Heat bullet
        energy_ratio = total_energy / (total_current + 1e-8)
        self._draw_bullet_chart(ax, 5.2, y_cursor - 0.3, 4.5, 0.35,
                               value=energy_ratio,
                               target=0.5,
                               ranges=[0.3, 0.5, 0.7],
                               label="E/(E+H)",
                               value_text=f"E:{total_energy:.0f} H:{total_heat:.0f}",
                               color_scheme='energy')
        
        y_cursor -= 1.0
        
        # ===== Section 2: Label Competition (Stacked bar) =====
        ax.text(0.2, y_cursor, "LABEL COMPETITION", fontsize=11, fontweight='bold', color='#2c3e50')
        y_cursor -= 0.3
        
        if self._label_indices:
            label_exc = excitation[self._label_indices].numpy()
            label_probs_np = probs[self._label_indices].numpy()
            
            # Stacked horizontal bar for label excitation
            self._draw_stacked_bar(ax, 0.3, y_cursor - 0.5, 9.4, 0.4, label_exc, 
                                   labels=[str(i) for i in range(10)])
            y_cursor -= 0.8
            
            # Winner indicator
            winner = np.argmax(label_probs_np)
            winner_prob = label_probs_np[winner]
            ax.text(0.3, y_cursor, f"Prediction: ", fontsize=9, color='#666')
            ax.text(1.8, y_cursor, f"{winner}", fontsize=14, fontweight='bold', 
                   color=f'C{winner}')
            ax.text(2.4, y_cursor, f" (p={winner_prob:.3f})", fontsize=9, color='#666')
            
            # Mini sparkline of winner's history
            if self._label_history["probs"]:
                winner_history = [h[winner] for h in self._label_history["probs"][-50:]]
                self._draw_sparkline(ax, 4.5, y_cursor - 0.15, 3.0, 0.3, winner_history, color=f'C{winner}')
        
        y_cursor -= 1.0
        
        # ===== Section 3: Entropy & Activity (Gauges) =====
        ax.text(0.2, y_cursor, "DYNAMICS", fontsize=11, fontweight='bold', color='#2c3e50')
        y_cursor -= 0.3
        
        # Entropy gauge
        entropy_val = self._history["entropy"][-1] if self._history["entropy"] else 0
        max_entropy = np.log2(len(self._label_indices) + 1)
        entropy_pct = entropy_val / max_entropy
        self._draw_gauge(ax, 1.5, y_cursor - 1.2, 1.0, entropy_pct, 
                        label="Entropy", value_text=f"{entropy_val:.2f}")
        
        # Activity gauge (new edges rate)
        avg_new_edges = np.mean(self._new_edges_history[-20:]) if self._new_edges_history else 0
        max_rate = max(self._new_edges_history) if self._new_edges_history else 1
        activity_pct = avg_new_edges / (max_rate + 1e-8)
        self._draw_gauge(ax, 5.0, y_cursor - 1.2, 1.0, activity_pct,
                        label="Bond Rate", value_text=f"{new_edges}/step")
        
        # Surprise from extra
        surprise = extra.get("surprise", 0) if extra else 0
        surprise_pct = min(1.0, surprise / 10.0)  # Normalize assuming max ~10
        self._draw_gauge(ax, 8.5, y_cursor - 1.2, 1.0, surprise_pct,
                        label="Surprise", value_text=f"{surprise:.2f}", invert=True)
        
        y_cursor -= 2.5
        
        # ===== Section 4: Time Series (Sparklines) =====
        ax.text(0.2, y_cursor, "HISTORY", fontsize=11, fontweight='bold', color='#2c3e50')
        y_cursor -= 0.3
        
        # Energy + Heat stacked area sparkline
        if len(self._history["total_energy"]) > 1:
            energy_hist = self._history["total_energy"][-100:]
            heat_hist = self._history["total_heat"][-100:]
            self._draw_area_sparkline(ax, 0.3, y_cursor - 0.6, 4.3, 0.5, 
                                      energy_hist, heat_hist, 
                                      label="Energy + Heat")
        
        # Loss and Perplexity sparklines (standard ML metrics)
        if len(self._history["loss"]) > 1:
            loss_hist = self._history["loss"][-100:]
            ppl_hist = self._history["ppl"][-100:]
            current_loss = loss_hist[-1] if loss_hist else 0.0
            current_ppl = ppl_hist[-1] if ppl_hist else 1.0
            self._draw_sparkline(ax, 5.2, y_cursor - 0.6, 4.3, 0.5, loss_hist, 
                                color='crimson', label=f"Loss: {current_loss:.3f}")
        elif len(self._history["num_edges"]) > 1:
            # Fallback to edges if no loss data
            edges_hist = self._history["num_edges"][-100:]
            self._draw_sparkline(ax, 5.2, y_cursor - 0.6, 4.3, 0.5, edges_hist, 
                                color='purple', label=f"Edges: {edges_hist[-1]}")
        
        y_cursor -= 1.0
        
        # Perplexity sparkline (below loss)
        if len(self._history["ppl"]) > 1:
            ppl_hist = self._history["ppl"][-100:]
            current_ppl = ppl_hist[-1] if ppl_hist else 1.0
            # Cap display at 1000 for readability
            ppl_display = [min(p, 1000) for p in ppl_hist]
            self._draw_sparkline(ax, 0.3, y_cursor - 0.6, 4.3, 0.5, ppl_display, 
                                color='darkorange', label=f"PPL: {current_ppl:.1f}")
            
            # Entropy sparkline 
            if len(self._history["entropy"]) > 1:
                entropy_hist = self._history["entropy"][-100:]
                current_entropy = entropy_hist[-1] if entropy_hist else 0.0
                self._draw_sparkline(ax, 5.2, y_cursor - 0.6, 4.3, 0.5, entropy_hist, 
                                    color='steelblue', label=f"Entropy: {current_entropy:.2f}")
            y_cursor -= 1.0
        
        # ===== Section 4.5: Chunks (if available) =====
        if self._chunk_history["num_chunks"]:
            ax.text(0.2, y_cursor, "CHUNKS", fontsize=11, fontweight='bold', color='#b8860b')
            y_cursor -= 0.3
            
            num_chunks = self._chunk_history["num_chunks"][-1]
            chunk_energy = self._chunk_history["chunk_energy"][-1]
            chunk_bonds = self._chunk_history["chunk_bonds"][-1]
            
            # Chunk count sparkline
            self._draw_sparkline(ax, 0.3, y_cursor - 0.5, 4.0, 0.4, 
                                self._chunk_history["num_chunks"][-100:],
                                color='gold', label=f"◆ Chunks: {num_chunks}")
            
            # Chunk excitation sparkline
            self._draw_sparkline(ax, 5.2, y_cursor - 0.5, 4.3, 0.4,
                                self._chunk_history["chunk_excitation"][-100:],
                                color='darkorange', label=f"Exc: {self._chunk_history['chunk_excitation'][-1]:.1f}")
            
            y_cursor -= 1.0
        
        # ===== Section 5: Key Metrics Summary =====
        ax.text(0.2, y_cursor, "SUMMARY", fontsize=11, fontweight='bold', color='#2c3e50')
        y_cursor -= 0.5
        
        # Build metrics list with standard ML metrics
        current_loss = self._history['loss'][-1] if self._history['loss'] else 0.0
        current_ppl = self._history['ppl'][-1] if self._history['ppl'] else 1.0
        
        metrics = [
            ("Step", f"{step:,}"),
            ("Loss", f"{current_loss:.3f}"),
            ("PPL", f"{current_ppl:.1f}"),
            ("Temp", f"{heat.mean().item():.1f}"),
            ("Err", f"{conservation_error:.2%}"),
        ]
        
        # Add chunk count if available
        if self._chunk_history["num_chunks"]:
            metrics.append(("Chunks", f"{self._chunk_history['num_chunks'][-1]}"))
        
        # Add distribution shape info from extra metrics
        if extra:
            num_active = extra.get("num_active_probs", 0)
            max_prob = extra.get("max_prob", 0)
            if num_active > 0:
                metrics.append(("Active", f"{int(num_active)}"))
            if max_prob > 0:
                metrics.append(("MaxP", f"{max_prob:.3f}"))
        
        for i, (name, val) in enumerate(metrics):
            x = 0.3 + (i % 7) * 1.4
            y_row = y_cursor if i < 7 else y_cursor - 0.7
            ax.text(x, y_row, name, fontsize=7, color='#888')
            ax.text(x, y_row - 0.3, val, fontsize=9, fontweight='bold', color='#2c3e50')
    
    def _draw_bullet_chart(self, ax, x, y, width, height, value, target, ranges, 
                           label, value_text, color_scheme='default'):
        """Draw a bullet chart."""
        from matplotlib.patches import Rectangle
        
        # Background ranges (light to dark)
        colors = ['#e0e0e0', '#c0c0c0', '#a0a0a0'] if color_scheme == 'default' else ['#ffe0e0', '#e0f0ff', '#e0ffe0']
        for i, r in enumerate(ranges):
            ax.add_patch(Rectangle((x, y), width * r, height, facecolor=colors[i], edgecolor='none'))
        
        # Value bar
        bar_color = '#2196F3' if color_scheme == 'default' else '#4CAF50'
        ax.add_patch(Rectangle((x, y + height*0.25), width * min(value, 1.0), height*0.5, 
                               facecolor=bar_color, edgecolor='none'))
        
        # Target marker
        ax.plot([x + width * target, x + width * target], [y, y + height], 
               'k-', linewidth=2)
        
        # Labels
        ax.text(x, y + height + 0.1, label, fontsize=8, color='#666')
        ax.text(x + width, y + height + 0.1, value_text, fontsize=8, color='#333', ha='right')
    
    def _draw_stacked_bar(self, ax, x, y, width, height, values, labels):
        """Draw a horizontal stacked bar chart."""
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        import numpy as np
        
        total = sum(values) + 1e-8
        colors = plt.cm.tab10(np.linspace(0, 1, len(values)))
        
        current_x = x
        for i, (val, label) in enumerate(zip(values, labels)):
            bar_width = (val / total) * width
            ax.add_patch(Rectangle((current_x, y), bar_width, height, 
                                   facecolor=colors[i], edgecolor='white', linewidth=0.5))
            if bar_width > 0.3:  # Only label if wide enough
                ax.text(current_x + bar_width/2, y + height/2, label, 
                       fontsize=7, ha='center', va='center', color='white', fontweight='bold')
            current_x += bar_width
    
    def _draw_gauge(self, ax, x, y, radius, value, label, value_text, invert=False):
        """Draw a semi-circular gauge."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        
        # Background arc
        bg = Wedge((x, y), radius, 180, 0, width=radius*0.3, 
                   facecolor='#e0e0e0', edgecolor='none')
        ax.add_patch(bg)
        
        # Value arc
        angle = 180 - (value * 180)  # 180 = 0%, 0 = 100%
        color = '#4CAF50' if not invert else plt.cm.RdYlGn(1 - value)
        if invert:
            color = plt.cm.RdYlGn_r(value)
        val_arc = Wedge((x, y), radius, angle, 180, width=radius*0.3,
                       facecolor=color, edgecolor='none')
        ax.add_patch(val_arc)
        
        # Labels
        ax.text(x, y - radius - 0.15, label, fontsize=8, ha='center', color='#666')
        ax.text(x, y - 0.1, value_text, fontsize=9, ha='center', fontweight='bold', color='#333')
    
    def _draw_sparkline(self, ax, x, y, width, height, values, color='steelblue', label=None):
        """Draw a simple sparkline."""
        import numpy as np
        
        if len(values) < 2:
            return
        
        values = np.array(values)
        xs = np.linspace(x, x + width, len(values))
        
        v_min, v_max = values.min(), values.max()
        if v_max - v_min < 1e-8:
            ys = np.full_like(values, y + height/2)
        else:
            ys = y + (values - v_min) / (v_max - v_min) * height
        
        ax.plot(xs, ys, color=color, linewidth=1.5)
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=20, zorder=5)
        
        if label:
            ax.text(x + width + 0.1, y + height/2, label, fontsize=7, va='center', color='#666')
    
    def _draw_area_sparkline(self, ax, x, y, width, height, values1, values2, label=None):
        """Draw a stacked area sparkline."""
        import numpy as np
        
        if len(values1) < 2:
            return
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        total = values1 + values2
        
        xs = np.linspace(x, x + width, len(values1))
        
        v_max = total.max()
        if v_max < 1e-8:
            return
        
        y1 = y + (values1 / v_max) * height
        y2 = y + (total / v_max) * height
        
        ax.fill_between(xs, y, y1, color='steelblue', alpha=0.6)
        ax.fill_between(xs, y1, y2, color='coral', alpha=0.6)
        ax.plot(xs, y2, color='darkred', linewidth=1)
        
        if label:
            ax.text(x + width + 0.1, y + height/2, label, fontsize=7, va='center', color='#666')
    
    def _render_3d_graph_advanced(self, energy: torch.Tensor, heat: torch.Tensor, 
                                   excitation: torch.Tensor, graph, step: int,
                                   max_nodes: int = 150, max_edges: int = 200) -> None:
        """Render advanced 3D bond graph with particles, bonds, heat cloud, and flow."""
        import numpy as np
        
        if self._positions_3d is None:
            self._compute_3d_positions()
        
        positions_3d = self._positions_3d
        energy_np = energy.numpy()
        heat_np = heat.numpy()
        exc_np = excitation.numpy()
        
        edge_src = graph.src.detach().cpu().numpy()
        edge_dst = graph.dst.detach().cpu().numpy()
        edge_weights = graph.w.detach().cpu().numpy()
        
        # Sample edges
        if len(edge_weights) > max_edges:
            edge_order = np.argsort(edge_weights)[::-1][:max_edges]
            edge_src = edge_src[edge_order]
            edge_dst = edge_dst[edge_order]
            edge_weights = edge_weights[edge_order]
        
        active_nodes = np.unique(np.concatenate([edge_src, edge_dst]))
        if len(active_nodes) > max_nodes:
            node_importance = energy_np[active_nodes] + exc_np[active_nodes]
            top_indices = np.argsort(node_importance)[::-1][:max_nodes]
            active_nodes = active_nodes[top_indices]
        
        active_set = set(active_nodes)
        ax = self._ax3d
        
        # Draw bonds with color gradient based on weight
        edge_weight_max = edge_weights.max() + 1e-8
        for i in range(0, len(edge_src), max(1, len(edge_src) // 80)):
            src, dst, w = edge_src[i], edge_dst[i], edge_weights[i]
            if src in active_set and dst in active_set:
                x = [positions_3d[src, 0], positions_3d[dst, 0]]
                y = [positions_3d[src, 1], positions_3d[dst, 1]]
                z = [positions_3d[src, 2], positions_3d[dst, 2]]
                
                # Color from blue (weak) to red (strong)
                strength = w / edge_weight_max
                color = (strength, 0.2, 1-strength, 0.2 + 0.6*strength)
                ax.plot(x, y, z, color=color, linewidth=0.5 + 2*strength)
        
        # Draw particles
        node_positions = positions_3d[active_nodes]
        node_energy = energy_np[active_nodes]
        node_heat = heat_np[active_nodes]
        node_exc = exc_np[active_nodes]
        
        # Size by excitation
        sizes = 15 + 120 * (node_exc / (node_exc.max() + 1e-8))
        
        # Color by energy (blue) with heat overlay
        total = node_energy + node_heat + 1e-8
        energy_ratio = node_energy / total  # 1 = all energy, 0 = all heat
        
        scatter = ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1], 
            node_positions[:, 2],
            c=energy_ratio,
            cmap='RdYlBu',  # Red=heat, Blue=energy
            s=sizes,
            alpha=0.85,
            edgecolors='gray',
            linewidths=0.3,
            vmin=0, vmax=1
        )
        
        # Heat cloud - large transparent red spheres
        heat_threshold = np.percentile(node_heat, 85)
        hot_mask = node_heat > heat_threshold
        if hot_mask.any():
            hot_positions = node_positions[hot_mask]
            hot_heat = node_heat[hot_mask]
            hot_sizes = 80 + 300 * (hot_heat / (hot_heat.max() + 1e-8))
            ax.scatter(
                hot_positions[:, 0],
                hot_positions[:, 1],
                hot_positions[:, 2],
                c='red',
                s=hot_sizes,
                alpha=0.15,
                edgecolors='none',
                marker='o'
            )
        
        # Label tokens as bright stars
        label_in_active = [n for n in active_nodes if n in self._label_indices]
        if label_in_active:
            label_positions = positions_3d[label_in_active]
            label_exc = exc_np[label_in_active]
            label_sizes = 200 + 400 * (label_exc / (label_exc.max() + 1e-8))
            ax.scatter(
                label_positions[:, 0],
                label_positions[:, 1],
                label_positions[:, 2],
                c='lime', s=label_sizes, marker='*', 
                edgecolors='darkgreen', linewidths=1.5, alpha=0.9
            )
            # Add labels
            for i, node_idx in enumerate(label_in_active):
                digit = self._label_indices.index(node_idx)
                ax.text(label_positions[i, 0], label_positions[i, 1], label_positions[i, 2],
                       f" {digit}", fontsize=8, color='darkgreen', fontweight='bold')
        
        ax.set_xlabel('PC1', fontsize=7)
        ax.set_ylabel('PC2', fontsize=7)
        ax.set_zlabel('PC3', fontsize=7)
        ax.set_title(f'Manifold: {len(active_nodes)} particles, {len(edge_src)} bonds\n'
                     f'Blue=energy, Red=heat, ★=labels', fontsize=8)
    
    def _render_3d_graph(self, energy: torch.Tensor, heat: torch.Tensor, 
                         excitation: torch.Tensor, graph, step: int,
                         max_nodes: int = 150, max_edges: int = 300) -> None:
        """Render real-time 3D bond graph with particles, bonds, and heat cloud."""
        # Redirect to advanced version
        self._render_3d_graph_advanced(energy, heat, excitation, graph, step, max_nodes, max_edges)

    def render_3d_graph(self, save_path: Optional[str] = None, max_nodes: int = 200, max_edges: int = 500):
        """Render a 3D visualization of the bond graph.
        
        Shows nodes as spheres (sized by energy, colored by heat) and
        bonds as lines (thickness by weight).
        
        Args:
            save_path: Optional path to save the figure
            max_nodes: Maximum nodes to display (samples if more)
            max_edges: Maximum edges to display (top by weight)
        """
        import matplotlib
        self._ensure_interactive_backend(matplotlib)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        graph = self.manifold.graph
        attractors = self.manifold.attractors
        
        if graph.num_edges == 0:
            print("No edges to visualize yet.")
            return None
        
        # Get node positions (use embedding positions, take first 3 dims)
        positions = attractors.get("position").detach().float().cpu().numpy()
        energy = attractors.get("energy").detach().float().cpu().numpy()
        heat = attractors.get("heat").detach().float().cpu().numpy()
        excitation = attractors.get("excitation").detach().float().cpu().numpy()
        
        # If positions are high-dimensional, project to 3D via PCA
        if positions.shape[1] > 3:
            # Simple PCA: center and take top 3 components
            positions_centered = positions - positions.mean(axis=0)
            try:
                u, s, vh = np.linalg.svd(positions_centered, full_matrices=False)
                positions_3d = u[:, :3] * s[:3]
            except Exception:
                # Fallback: just take first 3 dims
                positions_3d = positions[:, :3]
        else:
            positions_3d = positions[:, :3] if positions.shape[1] >= 3 else np.pad(positions, ((0,0), (0, 3-positions.shape[1])))
        
        # Get edges (top by weight)
        edge_src = graph.src.detach().cpu().numpy()
        edge_dst = graph.dst.detach().cpu().numpy()
        edge_weights = graph.w.detach().cpu().numpy()
        
        # Sort by weight and take top edges
        edge_order = np.argsort(edge_weights)[::-1][:max_edges]
        edge_src = edge_src[edge_order]
        edge_dst = edge_dst[edge_order]
        edge_weights = edge_weights[edge_order]
        
        # Find active nodes (nodes with edges)
        active_nodes = np.unique(np.concatenate([edge_src, edge_dst]))
        if len(active_nodes) > max_nodes:
            # Sample by energy
            node_importance = energy[active_nodes] + heat[active_nodes]
            top_indices = np.argsort(node_importance)[::-1][:max_nodes]
            active_nodes = active_nodes[top_indices]
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot edges first (so nodes are on top)
        edge_weight_max = edge_weights.max() + 1e-8
        for i in range(len(edge_src)):
            src, dst, w = edge_src[i], edge_dst[i], edge_weights[i]
            if src in active_nodes and dst in active_nodes:
                x = [positions_3d[src, 0], positions_3d[dst, 0]]
                y = [positions_3d[src, 1], positions_3d[dst, 1]]
                z = [positions_3d[src, 2], positions_3d[dst, 2]]
                alpha = 0.2 + 0.6 * (w / edge_weight_max)
                linewidth = 0.5 + 2 * (w / edge_weight_max)
                ax.plot(x, y, z, 'gray', alpha=alpha, linewidth=linewidth)
        
        # Plot nodes
        node_positions = positions_3d[active_nodes]
        node_energy = energy[active_nodes]
        node_heat = heat[active_nodes]
        
        # Size by total energy (energy + heat), color by heat ratio
        total = node_energy + node_heat + 1e-8
        sizes = 20 + 200 * (total / (total.max() + 1e-8))
        colors = node_heat / total  # 0 = all potential, 1 = all heat
        
        scatter = ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1], 
            node_positions[:, 2],
            c=colors, cmap='RdYlBu_r',  # Blue = energy, Red = heat
            s=sizes,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Highlight label nodes
        label_indices = [i for i, v in enumerate(self.vocab) if v.startswith("label_")]
        label_in_active = [n for n in active_nodes if n in label_indices]
        if label_in_active:
            label_positions = positions_3d[label_in_active]
            ax.scatter(
                label_positions[:, 0],
                label_positions[:, 1],
                label_positions[:, 2],
                c='lime', s=300, marker='*', edgecolors='black', linewidths=2,
                label='Label tokens'
            )
            # Add labels
            for i, node_idx in enumerate(label_in_active):
                label_name = self.vocab[node_idx] if node_idx < len(self.vocab) else f"?{node_idx}"
                ax.text(
                    label_positions[i, 0], 
                    label_positions[i, 1], 
                    label_positions[i, 2],
                    f"  {label_name}", fontsize=10, color='green', fontweight='bold'
                )
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Heat Fraction (blue=potential, red=dissipated)')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Bond Graph: {len(active_nodes)} nodes, {len(edge_src)} edges (top by weight)')
        
        if label_in_active:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D graph to {save_path}")
        else:
            plt.show()
        
        return fig

    def _ensure_interactive_backend(self, matplotlib_module) -> None:
        """Try to switch to an interactive backend if possible."""
        interactive_backends = {
            "MacOSX",
            "TkAgg",
            "QtAgg",
            "Qt5Agg",
            "GTK3Agg",
            "WXAgg",
        }
        current = matplotlib_module.get_backend()
        if current in interactive_backends:
            return

        for candidate in ("MacOSX", "TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib_module.use(candidate, force=True)
                return
            except Exception:
                continue

        if not self._warned_non_interactive:
            print(
                f"Dashboard: non-interactive matplotlib backend '{current}'. "
                "Real-time window may not appear."
            )
            self._warned_non_interactive = True
    
    def save(self, path: str) -> None:
        """Save current figure."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
    
    def close(self) -> None:
        """Close the figure."""
        import matplotlib.pyplot as plt
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
