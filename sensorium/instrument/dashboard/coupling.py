"""Mode-particle coupling visualization for ω-field dynamics.

Shows the interaction structure between modes and particles:
- Coupling strength heatmap
- Mode support distribution
- Anchor weight patterns
- Resonance network
"""

from __future__ import annotations

import numpy as np


class CouplingPlot:
    """Mode-particle coupling heatmap/network visualization.
    
    This visualization reveals the structure of how particles couple
    to ω-modes through the anchor mechanism:
    - Which particles support which modes
    - Coupling strength distribution
    - Mode isolation vs overlap
    - Emergent clustering
    """

    __slots__ = (
        'ax',
        '_initialized',
        '_last_coupling',
        '_im',
        '_ax_right',
        '_bars',
    )

    def __init__(self, ax) -> None:
        self.ax = ax
        self._initialized = False
        self._last_coupling = None
        # Heatmap image updated via set_data()
        self._im = ax.imshow(
            np.zeros((1, 1), dtype=np.float64),
            aspect='auto',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            interpolation='nearest',
        )
        ax.set_xlabel('Particle', fontsize=7)
        ax.set_ylabel('Mode', fontsize=7)

        # Right axis created once (do NOT create a new twinx each frame)
        self._ax_right = ax.twinx()
        self._ax_right.set_xlim(0, 1.5)
        self._ax_right.set_yticks([])
        self._ax_right.tick_params(labelsize=5)
        self._ax_right.set_xlabel('|Ψ|', fontsize=6)

        # Pre-create up to 16 bars; update widths/visibility each frame.
        self._bars = self._ax_right.barh(np.arange(16), np.zeros((16,), dtype=np.float64),
                                         height=0.8, color='#3498db', alpha=0.4)

    def update(self, state: dict) -> None:
        """Update coupling visualization from state dict."""
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v
        
        anchor_idx = get("mode_anchor_idx")
        anchor_weight = get("mode_anchor_weight")
        psi_amp = get("psi_amplitude")
        omega = get("omega_lattice")
        positions = get("positions")
        excitations = get("excitations")
        
        if anchor_idx is None or anchor_weight is None or psi_amp is None:
            self.ax.set_title("Mode coupling: awaiting data", fontsize=9)
            self._im.set_data(np.zeros((1, 1), dtype=np.float64))
            return
        
        amp = psi_amp.astype(np.float64)
        M = len(amp)
        
        # Infer slots per mode
        total_anchors = len(anchor_idx)
        slots = total_anchors // M if M > 0 else 0
        if slots == 0:
            self.ax.set_title("Mode coupling: no anchors", fontsize=9)
            self._im.set_data(np.zeros((1, 1), dtype=np.float64))
            return
        
        # Compute mode support (sum of anchor weights per mode)
        a_idx = anchor_idx.astype(np.int32)
        a_w = np.abs(anchor_weight.astype(np.float64))
        
        # Build coupling matrix (mode x particle)
        N = int(positions.shape[0]) if positions is not None else 0
        if N == 0:
            self.ax.set_title("Mode coupling: no particles", fontsize=9)
            self._im.set_data(np.zeros((1, 1), dtype=np.float64))
            return
        
        # Compute support per mode
        mode_support = np.zeros(M, dtype=np.float64)
        mode_particle_count = np.zeros(M, dtype=np.int32)
        
        for k in range(M):
            base = k * slots
            for s in range(slots):
                idx = int(a_idx[base + s])
                w = float(a_w[base + s])
                if 0 <= idx < N and w > 0:
                    mode_support[k] += w
                    mode_particle_count[k] += 1
        
        # Select top modes by amplitude for detailed view
        K = min(16, M)  # Show top 16 modes
        top_modes = np.argsort(amp)[-K:][::-1]
        
        # Build sparse coupling matrix for top modes
        coupling_matrix = np.zeros((K, min(N, 32)), dtype=np.float64)
        particle_map = {}  # Map global particle idx to display column
        col = 0
        
        for ki, k in enumerate(top_modes):
            base = k * slots
            for s in range(slots):
                idx = int(a_idx[base + s])
                w = float(a_w[base + s])
                if 0 <= idx < N and w > 0:
                    if idx not in particle_map:
                        if col < coupling_matrix.shape[1]:
                            particle_map[idx] = col
                            col += 1
                    if idx in particle_map:
                        coupling_matrix[ki, particle_map[idx]] = w
        
        # Trim to actual columns used
        if col > 0:
            coupling_matrix = coupling_matrix[:, :col]
        else:
            coupling_matrix = np.zeros((K, 1))
        
        # Normalize for display
        if np.max(coupling_matrix) > 0:
            coupling_matrix = coupling_matrix / np.max(coupling_matrix)
        
        # Display as heatmap
        self._im.set_data(coupling_matrix)
        
        # Y-axis: mode labels with omega values
        omega_arr = omega.astype(np.float64) if omega is not None else np.arange(M)
        y_labels = [f'ω={omega_arr[k]:.2f}' for k in top_modes]
        self.ax.set_yticks(np.arange(K))
        self.ax.set_yticklabels(y_labels, fontsize=5)
        
        # X-axis: particle indices
        if col > 0:
            x_ticks = np.arange(0, col, max(1, col // 8))
            self.ax.set_xticks(x_ticks)
            self.ax.set_xticklabels([str(i) for i in x_ticks], fontsize=5)

        # Update right-axis amplitude bars (widths + visibility)
        widths = amp[top_modes] / (np.max(amp) + 1e-8) if K > 0 else np.zeros((0,), dtype=np.float64)
        for i, rect in enumerate(self._bars):
            if i < K:
                rect.set_y(i - 0.4)
                rect.set_height(0.8)
                rect.set_width(float(widths[i]))
                rect.set_visible(True)
            else:
                rect.set_visible(False)
        
        # Compute and display coupling entropy (diversity measure)
        # Higher entropy = more distributed coupling
        coupling_sums = np.sum(coupling_matrix, axis=1)
        if np.sum(coupling_sums) > 0:
            p = coupling_sums / np.sum(coupling_sums)
            p = p[p > 0]
            coupling_entropy = float(-np.sum(p * np.log(p))) / np.log(len(p) + 1)
        else:
            coupling_entropy = 0.0
        
        total_support = float(np.sum(mode_support))
        avg_connections = float(np.mean(mode_particle_count[top_modes]))
        
        self.ax.set_title(
            f'Mode-particle coupling | '
            f'H_c={coupling_entropy:.2f} | '
            f'⟨conn⟩={avg_connections:.1f} | '
            f'support={total_support:.1f}',
            fontsize=8, pad=4
        )
